// src/training/lr_l1.rs — L1-regularized logistic regression baseline.
//
// Class convention: upstream labels are 0.0 = UP, 1.0 = DOWN (LightGBM-
// internal; see retrain::build_training_set). This module works in
// UP-positive space: y = 1.0 iff the row is UP, so the model's raw output
// is P(UP), matching predict.rs's `probs[0]` slot. `BetaCal` is applied
// downstream by the unchanged `inference::predict::predict_one`.

use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::calibration::{self, BetaCal};
use crate::feature_engine::TOTAL_FEATURES;
use crate::retrain::purged_kfold_indices;

/// Internal LR blob layout version. Bump only if the `LrL1Model` struct layout changes.
const LR_BLOB_FORMAT_TAG: u16 = 2;

/// Inverse-regularization-strength grid for inner-CV `C` search (`C` = sklearn convention: larger = weaker L1).
pub const C_GRID: [f64; 4] = [0.01, 0.1, 1.0, 10.0];

/// Inner k for `C`-selection CV.
const INNER_K: usize = 3;

/// Coordinate-descent iteration limits and convergence tolerance.
const MAX_OUTER_ITERS: usize = 50;
const MAX_INNER_ITERS: usize = 100;
const CD_TOL: f64 = 1e-4;

/// Persisted L1-logistic model. Serialized as JSON into `models.model_bytes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrL1Model {
    /// = `LR_BLOB_FORMAT_TAG`. Guards against struct-layout drift.
    pub format_tag: u16,
    /// Coefficients in *standardized* space, length `TOTAL_FEATURES`.
    pub coefficients: Vec<f64>,
    /// Intercept in standardized space.
    pub intercept: f64,
    /// Per-column NaN-replacement means, fit on training rows only, length `TOTAL_FEATURES`.
    pub impute_means: Vec<f64>,
    /// Per-column mean of the imputed training matrix, length `TOTAL_FEATURES`.
    pub scaler_mean: Vec<f64>,
    /// Per-column std of the imputed training matrix (floored 1e-9), length `TOTAL_FEATURES`.
    pub scaler_std: Vec<f64>,
    /// 3-parameter beta calibration: `σ(a·log(p_up) + b·log(1−p_up) + c)`.
    /// `IDENTITY = (1.0, -1.0, 0.0)` for CV-fold models or degenerate val slices.
    pub beta_a: f64,
    pub beta_b: f64,
    pub beta_c: f64,
    /// Inverse regularization strength selected by inner CV -- diagnostic.
    pub best_c: f64,
}

/// Legacy blob shape for v14/v15 JSON blobs (with `platt_a, platt_b` fields).
/// Deserialized by `LrL1Model::from_bytes` and converted to BetaCal in memory.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct LrL1ModelV15Legacy {
    format_tag: u16,
    coefficients: Vec<f64>,
    intercept: f64,
    impute_means: Vec<f64>,
    scaler_mean: Vec<f64>,
    scaler_std: Vec<f64>,
    platt_a: f64,
    platt_b: f64,
    best_c: f64,
}

impl LrL1Model {
    /// Serialize to the JSON blob stored in `models.model_bytes`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| anyhow!("LrL1Model serialize: {e}"))
    }

    /// Parse a JSON blob, validating the format tag and vector lengths.
    /// v16+ blobs use the current layout; v14/v15 blobs are deserialized via
    /// `LrL1ModelV15Legacy` and converted to BetaCal via `calibration::platt_to_beta`.
    pub fn from_bytes(bytes: &[u8], format_version: i16) -> Result<Self> {
        let m = if format_version >= 16 {
            let m: LrL1Model = serde_json::from_slice(bytes)
                .map_err(|e| anyhow!("LrL1Model v16 deserialize: {e}"))?;
            if m.format_tag != LR_BLOB_FORMAT_TAG {
                return Err(anyhow!(
                    "LrL1Model v{format_version} format_tag {} != current {}",
                    m.format_tag,
                    LR_BLOB_FORMAT_TAG
                ));
            }
            m
        } else {
            let legacy: LrL1ModelV15Legacy = serde_json::from_slice(bytes)
                .map_err(|e| anyhow!("LrL1ModelV15Legacy deserialize: {e}"))?;
            log_legacy_conversion_once(format_version);
            let beta = calibration::platt_to_beta(legacy.platt_a, legacy.platt_b);
            LrL1Model {
                format_tag: LR_BLOB_FORMAT_TAG,
                coefficients: legacy.coefficients,
                intercept: legacy.intercept,
                impute_means: legacy.impute_means,
                scaler_mean: legacy.scaler_mean,
                scaler_std: legacy.scaler_std,
                beta_a: beta.a,
                beta_b: beta.b,
                beta_c: beta.c,
                best_c: legacy.best_c,
            }
        };
        for (name, v) in [
            ("coefficients", &m.coefficients),
            ("impute_means", &m.impute_means),
            ("scaler_mean", &m.scaler_mean),
            ("scaler_std", &m.scaler_std),
        ] {
            if v.len() != TOTAL_FEATURES {
                return Err(anyhow!(
                    "LrL1Model.{name} len {} != TOTAL_FEATURES {}",
                    v.len(),
                    TOTAL_FEATURES
                ));
            }
        }
        Ok(m)
    }

    /// `BetaCal { a, b, c }` for the downstream `predict_one` calibration step.
    pub fn beta(&self) -> BetaCal {
        BetaCal {
            a: self.beta_a,
            b: self.beta_b,
            c: self.beta_c,
        }
    }

    /// `|coefficient|` per feature, length `TOTAL_FEATURES`.
    pub fn abs_coefficients(&self) -> Vec<f64> {
        self.coefficients.iter().map(|c| c.abs()).collect()
    }

    /// Count of non-zero coefficients -- the L1 sparsity headline.
    pub fn nonzero_coef_count(&self) -> usize {
        self.coefficients.iter().filter(|c| **c != 0.0).count()
    }

    /// Linear score `intercept + w · standardize(impute(row))`.
    fn linear_score(&self, raw_at: impl Fn(usize) -> f64) -> f64 {
        let mut eta = self.intercept;
        for j in 0..TOTAL_FEATURES {
            let raw = raw_at(j);
            let imputed = if raw.is_nan() { self.impute_means[j] } else { raw };
            let std = self.scaler_std[j];
            let x = if std > 0.0 {
                (imputed - self.scaler_mean[j]) / std
            } else {
                0.0
            };
            eta += self.coefficients[j] * x;
        }
        eta
    }

    /// Raw P(UP) for one flat `f64` feature row.
    fn predict_up_row(&self, row: &[f64]) -> f64 {
        debug_assert_eq!(row.len(), TOTAL_FEATURES);
        sigmoid(self.linear_score(|j| row[j]))
    }

    /// Raw P(UP) for the live inference path (f32 feature array). BetaCal is applied downstream.
    pub fn predict_up(&self, features: &[f32; TOTAL_FEATURES]) -> f64 {
        sigmoid(self.linear_score(|j| features[j] as f64))
    }
}

/// Warn once per process when a legacy v14/v15 blob is converted to BetaCal at load.
fn log_legacy_conversion_once(format_version: i16) {
    static WARNED: OnceLock<()> = OnceLock::new();
    if WARNED.set(()).is_ok() {
        warn!(
            "[LrL1Model] converted v{} legacy Platt calibration to BetaCal at load — \
             v14/v15 blobs are read-only; new retrains promote v16.",
            format_version
        );
    }
}

/// Numerically stable logistic sigmoid.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// UP-positive target: 1.0 iff the upstream label is UP (0.0).
fn up_target(label: f32) -> f64 {
    if label == 0.0 {
        1.0
    } else {
        0.0
    }
}

/// One-sample binary cross-entropy with probability clamping.
fn binary_logloss_one(p: f64, y: f64) -> f64 {
    let eps = 1e-15;
    let pc = p.clamp(eps, 1.0 - eps);
    -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln())
}

/// Train an L1-logistic model; `C` chosen by inner 3-fold CV. BetaCal left at identity.
pub fn train_lr_l1(feats: &[f64], labels: &[f32]) -> Result<LrL1Model> {
    train_lr_l1_with_weights_opt(feats, labels, None)
}

/// Weighted training. Weights normalized to `Σwᵢ = n` internally.
pub fn train_lr_l1_with_weights(
    feats: &[f64],
    labels: &[f32],
    weights: &[f32],
) -> Result<LrL1Model> {
    train_lr_l1_with_weights_opt(feats, labels, Some(weights))
}

fn train_lr_l1_with_weights_opt(
    feats: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
) -> Result<LrL1Model> {
    // Convert + normalize once; both select_c and fit_core consume the
    // same normalized vec.
    let sw = match weights {
        None => None,
        Some(w) => {
            if w.len() != labels.len() {
                return Err(anyhow!(
                    "train_lr_l1_with_weights: weights len {} != labels len {}",
                    w.len(),
                    labels.len()
                ));
            }
            Some(normalize_sample_weights(w))
        }
    };
    let best_c = select_c(feats, labels, sw.as_deref());
    fit_core(feats, labels, best_c, sw.as_deref())
}

/// Normalize sample weights so `Σwᵢ = n`. Falls back to uniform on degenerate input.
fn normalize_sample_weights(w: &[f32]) -> Vec<f64> {
    let n = w.len();
    if n == 0 {
        return Vec::new();
    }
    let sum: f64 = w.iter().map(|&x| x as f64).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0; n];
    }
    let scale = n as f64 / sum;
    w.iter().map(|&x| x as f64 * scale).collect()
}

/// Train the promoted L1-logistic model; fits BetaCal on the held-out val slice.
pub fn train_lr_l1_promoted(
    train_feats: &[f64],
    train_labels: &[f32],
    val_feats: &[f64],
    val_labels: &[f32],
) -> Result<LrL1Model> {
    train_lr_l1_promoted_with_weights_opt(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        None,
    )
}

/// Weighted promoted-LR-L1 training. Weights apply to training fit only; val calibration is unweighted.
pub fn train_lr_l1_promoted_with_weights(
    train_feats: &[f64],
    train_labels: &[f32],
    val_feats: &[f64],
    val_labels: &[f32],
    train_weights: &[f32],
) -> Result<LrL1Model> {
    train_lr_l1_promoted_with_weights_opt(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        Some(train_weights),
    )
}

fn train_lr_l1_promoted_with_weights_opt(
    train_feats: &[f64],
    train_labels: &[f32],
    val_feats: &[f64],
    val_labels: &[f32],
    train_weights: Option<&[f32]>,
) -> Result<LrL1Model> {
    let mut model = train_lr_l1_with_weights_opt(train_feats, train_labels, train_weights)?;
    if !val_labels.is_empty() {
        let n = val_labels.len();
        let mut p_up = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            p_up.push(
                model.predict_up_row(&val_feats[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]),
            );
            y.push(up_target(val_labels[i]));
        }
        let cal = calibration::fit_beta_calibration(&p_up, &y);
        model.beta_a = cal.a;
        model.beta_b = cal.b;
        model.beta_c = cal.c;
    }
    Ok(model)
}

/// Argmax accuracy of an LR-L1 model on a flat val buffer.
pub fn lr_l1_accuracy(model: &LrL1Model, feats: &[f64], labels: &[f32]) -> f64 {
    let n = labels.len();
    if n == 0 {
        return 0.0;
    }
    let mut correct = 0usize;
    for i in 0..n {
        let p_up = model.predict_up_row(&feats[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
        // class 0 = UP, class 1 = DOWN; predict UP iff p_up >= 0.5.
        let pred: f32 = if p_up >= 0.5 { 0.0 } else { 1.0 };
        if (pred - labels[i]).abs() < f32::EPSILON {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

/// Mean binary log-loss of an LR-L1 model on a flat buffer.
pub fn lr_l1_binary_logloss(model: &LrL1Model, feats: &[f64], labels: &[f32]) -> f64 {
    let n = labels.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let p_up = model.predict_up_row(&feats[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
        sum += binary_logloss_one(p_up, up_target(labels[i]));
    }
    sum / n as f64
}

/// Raw P(UP) as `f32` for one live feature row.
pub fn predict_lr_l1(model: &LrL1Model, row: &[f32; TOTAL_FEATURES]) -> f32 {
    model.predict_up(row) as f32
}

/// impute -> standardize -> coordinate-descent L1 logistic. Returns a model with identity BetaCal.
fn fit_core(
    feats: &[f64],
    labels: &[f32],
    c: f64,
    sample_weights: Option<&[f64]>,
) -> Result<LrL1Model> {
    let n = labels.len();
    if n == 0 {
        return Err(anyhow!("fit_core: empty training set"));
    }
    if feats.len() != n * TOTAL_FEATURES {
        return Err(anyhow!(
            "fit_core: feats len {} != n {} x TOTAL_FEATURES {}",
            feats.len(),
            n,
            TOTAL_FEATURES
        ));
    }
    if let Some(sw) = sample_weights {
        if sw.len() != n {
            return Err(anyhow!(
                "fit_core: sample_weights len {} != n {}",
                sw.len(),
                n
            ));
        }
    }

    // Imputer: per-column mean ignoring NaN; all-NaN column gets 0.0.
    let mut impute_means = vec![0.0_f64; TOTAL_FEATURES];
    for j in 0..TOTAL_FEATURES {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for i in 0..n {
            let v = feats[i * TOTAL_FEATURES + j];
            if !v.is_nan() {
                sum += v;
                cnt += 1;
            }
        }
        impute_means[j] = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
    }

    let mut imputed = vec![0.0_f64; n * TOTAL_FEATURES];
    for i in 0..n {
        for j in 0..TOTAL_FEATURES {
            let v = feats[i * TOTAL_FEATURES + j];
            imputed[i * TOTAL_FEATURES + j] = if v.is_nan() { impute_means[j] } else { v };
        }
    }

    // Fresh z-score scaler on the imputed matrix; std floored at 1e-9.
    let mut scaler_mean = vec![0.0_f64; TOTAL_FEATURES];
    let mut scaler_std = vec![0.0_f64; TOTAL_FEATURES];
    for j in 0..TOTAL_FEATURES {
        let mut sum = 0.0;
        for i in 0..n {
            sum += imputed[i * TOTAL_FEATURES + j];
        }
        let mean = sum / n as f64;
        let mut ss = 0.0;
        for i in 0..n {
            let d = imputed[i * TOTAL_FEATURES + j] - mean;
            ss += d * d;
        }
        scaler_mean[j] = mean;
        scaler_std[j] = (ss / n as f64).sqrt().max(1e-9);
    }

    // Standardized design matrix.
    let mut x = vec![0.0_f64; n * TOTAL_FEATURES];
    for i in 0..n {
        for j in 0..TOTAL_FEATURES {
            x[i * TOTAL_FEATURES + j] =
                (imputed[i * TOTAL_FEATURES + j] - scaler_mean[j]) / scaler_std[j];
        }
    }

    let y: Vec<f64> = labels.iter().map(|&l| up_target(l)).collect();
    let (coefficients, intercept) = coordinate_descent(&x, &y, n, c, sample_weights);

    Ok(LrL1Model {
        format_tag: LR_BLOB_FORMAT_TAG,
        coefficients,
        intercept,
        impute_means,
        scaler_mean,
        scaler_std,
        beta_a: calibration::IDENTITY.a,
        beta_b: calibration::IDENTITY.b,
        beta_c: calibration::IDENTITY.c,
        best_c: c,
    })
}

/// Pick `C` from `C_GRID` by inner k-fold CV, scoring mean binary log-loss.
fn select_c(feats: &[f64], labels: &[f32], sample_weights: Option<&[f64]>) -> f64 {
    let folds = purged_kfold_indices(labels.len(), INNER_K, 0);
    let mut best_c = 1.0_f64;
    let mut best_loss = f64::INFINITY;
    for &c in C_GRID.iter() {
        let mut loss_sum = 0.0_f64;
        let mut scored = 0usize;
        for (train_idx, val_idx) in folds.iter() {
            if val_idx.is_empty() || train_idx.is_empty() {
                continue;
            }
            let (tf, tl) = gather(feats, labels, train_idx);
            let tw: Option<Vec<f64>> = sample_weights.map(|sw| {
                train_idx.iter().map(|&i| sw[i]).collect()
            });
            let model = match fit_core(&tf, &tl, c, tw.as_deref()) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let mut fold_loss = 0.0_f64;
            for &i in val_idx {
                let p = model
                    .predict_up_row(&feats[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
                fold_loss += binary_logloss_one(p, up_target(labels[i]));
            }
            loss_sum += fold_loss / val_idx.len() as f64;
            scored += 1;
        }
        if scored > 0 {
            let mean_loss = loss_sum / scored as f64;
            if mean_loss < best_loss {
                best_loss = mean_loss;
                best_c = c;
            }
        }
    }
    best_c
}

/// Gather a subset of rows (by index) into a contiguous flat buffer.
fn gather(feats: &[f64], labels: &[f32], idx: &[usize]) -> (Vec<f64>, Vec<f32>) {
    let mut f = Vec::with_capacity(idx.len() * TOTAL_FEATURES);
    let mut l = Vec::with_capacity(idx.len());
    for &i in idx {
        f.extend_from_slice(&feats[i * TOTAL_FEATURES..(i + 1) * TOTAL_FEATURES]);
        l.push(labels[i]);
    }
    (f, l)
}

/// Soft-thresholding operator `S(a, lambda) = sign(a) * max(|a| - lambda, 0)`.
fn soft_threshold(a: f64, lambda: f64) -> f64 {
    if a > lambda {
        a - lambda
    } else if a < -lambda {
        a + lambda
    } else {
        0.0
    }
}

/// Cyclic coordinate descent for L1-penalized logistic regression (GLMNet).
/// Minimizes `(1/n) Σ logloss_i + (1/C) ||w||_1`; intercept unpenalized.
/// Outer loop: IRLS re-linearization. Inner loop: cyclic soft-thresholding.
fn coordinate_descent(
    x: &[f64],
    y: &[f64],
    n: usize,
    c: f64,
    sample_weights: Option<&[f64]>,
) -> (Vec<f64>, f64) {
    let p = TOTAL_FEATURES;
    let threshold = 1.0 / c; // soft-threshold parameter
    let mut w = vec![0.0_f64; p];
    // Intercept initialized to the empirical log-odds.
    let ybar = if let Some(sw) = sample_weights {
        let num: f64 = y.iter().zip(sw.iter()).map(|(&yi, &swi)| yi * swi).sum();
        let den: f64 = sw.iter().sum::<f64>().max(1e-12);
        (num / den).clamp(1e-6, 1.0 - 1e-6)
    } else {
        (y.iter().sum::<f64>() / n as f64).clamp(1e-6, 1.0 - 1e-6)
    };
    let mut b = (ybar / (1.0 - ybar)).ln();

    for _outer in 0..MAX_OUTER_ITERS {
        let w_before = w.clone();
        let b_before = b;

        // IRLS: working weight `v_i = sw_i * p_i(1-p_i)`, residual `r_i = (y_i - p_i) / p_i(1-p_i)`.
        let mut v = vec![0.0_f64; n];
        let mut r = vec![0.0_f64; n];
        for i in 0..n {
            let mut eta = b;
            for j in 0..p {
                eta += w[j] * x[i * p + j];
            }
            let pr = sigmoid(eta);
            let vi_irls = (pr * (1.0 - pr)).max(1e-5);
            let sw_i = sample_weights.map(|sw| sw[i]).unwrap_or(1.0);
            v[i] = sw_i * vi_irls;
            r[i] = (y[i] - pr) / vi_irls;
        }

        for _inner in 0..MAX_INNER_ITERS {
            let mut inner_max = 0.0_f64;

            // Unpenalized intercept update.
            let sum_v: f64 = v.iter().sum(); // intercept column is all-ones
            if sum_v > 1e-12 {
                let db =
                    v.iter().zip(r.iter()).map(|(&vi, &ri)| vi * ri).sum::<f64>() / sum_v;
                b += db;
                for ri in r.iter_mut() {
                    *ri -= db;
                }
                inner_max = inner_max.max(db.abs());
            }

            // Coordinate updates with soft-thresholding.
            for j in 0..p {
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                for i in 0..n {
                    let xij = x[i * p + j];
                    let rj = r[i] + w[j] * xij; // partial residual (add j back in)
                    num += v[i] * xij * rj;
                    den += v[i] * xij * xij;
                }
                let den = den.max(1e-12);
                let wj_new = soft_threshold(num, threshold) / den;
                let dw = wj_new - w[j];
                if dw != 0.0 {
                    for i in 0..n {
                        r[i] -= dw * x[i * p + j];
                    }
                    w[j] = wj_new;
                    inner_max = inner_max.max(dw.abs());
                }
            }

            if inner_max < CD_TOL {
                break;
            }
        }

        let mut outer_max = (b - b_before).abs();
        for j in 0..p {
            outer_max = outer_max.max((w[j] - w_before[j]).abs());
        }
        if outer_max < CD_TOL {
            break;
        }
    }

    (w, b)
}

