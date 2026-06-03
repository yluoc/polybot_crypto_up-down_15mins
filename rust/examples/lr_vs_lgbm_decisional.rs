// examples/lr_vs_lgbm_decisional.rs — LR-L1 vs LightGBM under purged k-fold + Wilson Gate 4.
// Read-only: no model promotion, no DB writes.
//
// Run:  DATABASE_URL=postgres://... cargo run --example \
//           lr_vs_lgbm_decisional -- [lookback_days]   (default 30)

use anyhow::{anyhow, Result};
use polybot::db;
use polybot::feature_engine::TOTAL_FEATURES;
use polybot::retrain::{
    class_counts, effective_sample_size, evaluate_pooled_cv, exponential_decay_weights,
    purged_kfold_indices, train_booster, validation_accuracy, wilson_lower_bound_one_sided,
    CV_EMBARGO_ROWS, CV_K_FOLDS, CV_MIN_FOLDS_PASSING, GATE4_BASELINE_MARGIN, MIN_ROWS_FOR_GATE,
    SAMPLE_WEIGHT_HALF_LIFE_DAYS, WILSON_ALPHA,
};
use polybot::signal::Symbol;
use polybot::training::lr_l1::{lr_l1_accuracy, train_lr_l1_with_weights};
use polybot::training::offline::load_offline_training_sets;

/// Number of boosting iterations cap — matches `RetrainArgs::trees` default.
const TREES: u32 = 500;

/// Median of a slice — empty → NaN (so the table shows it clearly).
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// Per-family fold tallies for the report.
struct FamilyReport {
    val_accs: Vec<f64>,
    wilson_lbs: Vec<f64>,
    folds_passing: usize,
}

impl FamilyReport {
    fn new() -> Self {
        FamilyReport {
            val_accs: Vec::new(),
            wilson_lbs: Vec::new(),
            folds_passing: 0,
        }
    }
    fn passed_gate4(&self) -> bool {
        self.folds_passing >= CV_MIN_FOLDS_PASSING
    }
    fn print_row(&self, model: &str) {
        let decision = if self.passed_gate4() { "pass" } else { "reject" };
        println!(
            "  {:<8} | {:<14.4} | {:<16.4} | {}/{:<11} | {}",
            model,
            median(&self.val_accs),
            median(&self.wilson_lbs),
            self.folds_passing,
            CV_K_FOLDS,
            decision
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let lookback_days: i64 = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(30);
    let database_url =
        std::env::var("DATABASE_URL").map_err(|_| anyhow!("DATABASE_URL must be set"))?;
    let pool = db::init_pool(&database_url).await?;

    let sets = load_offline_training_sets(&pool, lookback_days).await?;

    let half_life_days = SAMPLE_WEIGHT_HALF_LIFE_DAYS;
    println!(
        "\n=== LR-L1 vs LightGBM decisional baseline ({lookback_days}d window, \
         sample_weight_half_life={half_life_days}d) ==="
    );

    for data in &sets {
        let short = &data.symbol_short;
        let features_flat = &data.features_flat;
        let labels = &data.labels;
        let timestamps_ms = &data.timestamps_ms;
        let n_rows = data.n_rows();

        println!("\nSymbol: {short}");
        if n_rows < MIN_ROWS_FOR_GATE {
            println!(
                "  only {n_rows} labelled rows (need >= {MIN_ROWS_FOR_GATE}) — Gate 4 not evaluable"
            );
            continue;
        }
        let full_weights = exponential_decay_weights(timestamps_ms, half_life_days);
        let full_ess = effective_sample_size(&full_weights);
        println!(
            "  labelled rows: {n_rows}  (full-set ESS={full_ess:.0}, half_life={half_life_days}d)"
        );
        println!("  Model    | Median val_acc | Median Wilson LB | Folds passing | Decision");

        // Purged 5-fold CV, both families on identical folds.
        let folds = purged_kfold_indices(n_rows, CV_K_FOLDS, CV_EMBARGO_ROWS);
        let mut lgbm = FamilyReport::new();
        let mut lr = FamilyReport::new();

        for (train_idx, val_idx) in folds.iter() {
            let mut tf = Vec::with_capacity(train_idx.len() * TOTAL_FEATURES);
            let mut tl = Vec::with_capacity(train_idx.len());
            let mut tw = Vec::with_capacity(train_idx.len());
            for &i in train_idx {
                tf.extend_from_slice(&features_flat[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
                tl.push(labels[i]);
                tw.push(full_weights[i]);
            }
            let mut vf = Vec::with_capacity(val_idx.len() * TOTAL_FEATURES);
            let mut vl = Vec::with_capacity(val_idx.len());
            for &i in val_idx {
                vf.extend_from_slice(&features_flat[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
                vl.push(labels[i]);
            }

            let n_val = vl.len() as u32;
            let val_counts = class_counts(&vl);
            let fold_baseline = (val_counts[0].max(val_counts[1])) as f64 / n_val.max(1) as f64;
            let score = |acc: f64, report: &mut FamilyReport| {
                let successes = (acc * n_val as f64).round() as u32;
                let wlb = wilson_lower_bound_one_sided(successes, n_val, WILSON_ALPHA);
                if wlb > fold_baseline + GATE4_BASELINE_MARGIN {
                    report.folds_passing += 1;
                }
                report.val_accs.push(acc);
                report.wilson_lbs.push(wlb);
            };

            match train_booster(&tf, &tl, &vf, &vl, TREES, Some(&tw))
                .and_then(|b| validation_accuracy(&b, &vf, &vl))
            {
                Ok(acc) => score(acc, &mut lgbm),
                Err(e) => eprintln!("  [warn] {short}: lightgbm fold failed: {e}"),
            }
            match train_lr_l1_with_weights(&tf, &tl, &tw) {
                Ok(m) => score(lr_l1_accuracy(&m, &vf, &vl), &mut lr),
                Err(e) => eprintln!("  [warn] {short}: lr_l1 fold failed: {e}"),
            }
        }

        lgbm.print_row("lightgbm");
        lr.print_row("lr_l1");

        // Q5 winner: clears Gate 4 => candidate; both => higher median.
        let lgbm_med = lgbm.passed_gate4().then(|| median(&lgbm.val_accs));
        let lr_med = lr.passed_gate4().then(|| median(&lr.val_accs));
        let winner = match (lgbm_med, lr_med) {
            (None, None) => "neither (both reject Gate 4)".to_string(),
            (Some(_), None) => "lightgbm".to_string(),
            (None, Some(_)) => "lr_l1".to_string(),
            (Some(a), Some(b)) => {
                if b > a {
                    "lr_l1 (higher median val_acc among passers)".to_string()
                } else {
                    "lightgbm (higher median val_acc among passers)".to_string()
                }
            }
        };
        println!("  -> winner: {winner}");
    }

    // Pooled multi-task LightGBM comparison: skipped unless every active symbol
    // has >= MIN_ROWS_FOR_GATE labelled rows.
    let pooled_eligible: Vec<&polybot::training::offline::OfflineSymbolData> = sets
        .iter()
        .filter(|d| d.n_rows() >= MIN_ROWS_FOR_GATE)
        .collect();
    if pooled_eligible.len() == sets.len() && !sets.is_empty() {
        println!("\n=== Pooled multi-task LightGBM (v15) ===");
        let per_symbol: Vec<(Symbol, Vec<f64>, Vec<f32>, Vec<i64>)> = pooled_eligible
            .iter()
            .map(|d| {
                let sym = Symbol::from_str_ci(&d.symbol_short).expect("active symbol parses");
                (
                    sym,
                    d.features_flat.clone(),
                    d.labels.clone(),
                    d.timestamps_ms.clone(),
                )
            })
            .collect();
        match evaluate_pooled_cv(per_symbol, TREES, half_life_days) {
            Err(e) => println!("  pooled CV failed: {e:#}"),
            Ok(result) => {
                println!(
                    "  symbol_id | pooled median | baseline median | pooled-vs-baseline | wilson pass folds"
                );
                for s in 0..result.per_sym.len() {
                    let e = &result.per_sym[s];
                    let delta = e.pooled_median_val_acc - e.baseline_median_val_acc;
                    let decision = if e.pooled_folds_wilson_pass >= CV_MIN_FOLDS_PASSING
                        && delta >= 0.0
                    {
                        "pass"
                    } else {
                        "reject"
                    };
                    println!(
                        "  {:^9} | {:^13.4} | {:^15.4} | {:^+18.4} | {}/{:<5} {}",
                        s,
                        e.pooled_median_val_acc,
                        e.baseline_median_val_acc,
                        delta,
                        e.pooled_folds_wilson_pass,
                        CV_K_FOLDS,
                        decision
                    );
                }
            }
        }
    } else {
        println!(
            "\n=== Pooled multi-task LightGBM (v15) — SKIPPED ===\n  \
             only {}/{} symbols have >= {} rows; pooled is all-or-nothing per design",
            pooled_eligible.len(),
            sets.len(),
            MIN_ROWS_FOR_GATE
        );
    }

    println!();
    Ok(())
}
