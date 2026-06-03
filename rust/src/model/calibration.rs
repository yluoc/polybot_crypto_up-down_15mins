// src/calibration.rs
//
// Three-parameter beta calibration `σ(a·log(p) + b·log(1−p) + c)`.
// Single source of truth for fit, apply, and serialisation of calibration params.
// v15 blobs (Platt 2-param) are converted to BetaCal at load via `platt_to_beta`.

use std::f64::consts::E;

/// Three-parameter beta calibration `(a, b, c)`. Live-applied as
/// `σ(a · log(p) + b · log(1−p) + c)` to raw P(UP).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaCal {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

/// Identity calibration `(1.0, −1.0, 0.0)` — no-op; equals `σ(logit(p)) = p`.
pub const IDENTITY: BetaCal = BetaCal { a: 1.0, b: -1.0, c: 0.0 };

/// Convert v15 Platt `(a, b)` to the equivalent `BetaCal { a, b: -a, c: b }`.
/// Information-preserving: `σ(a·logit(p)+b) ≡ σ(a·log(p)−a·log(1−p)+b)`.
pub fn platt_to_beta(a: f64, b: f64) -> BetaCal {
    BetaCal { a, b: -a, c: b }
}

// Trailer magic bytes and lengths for v16 beta calibration.
pub const BETA_MAGIC: &[u8; 4] = b"BETA";
pub const BETA_POOLED_MAGIC: &[u8; 4] = b"BPCL";

/// Per-symbol trailer: 3 f64 + 4-byte magic = 28 bytes.
pub const BETA_TRAILER_LEN: usize = 8 * 3 + 4;

/// Pooled trailer: 3 f64 × INSTRUMENT_COUNT + 4-byte magic.
pub const BETA_POOLED_TRAILER_LEN: usize =
    8 * 3 * crate::feature_engine::INSTRUMENT_COUNT + 4;

// Clamp bounds before log(p) / log(1−p).
const CLAMP_EPS: f64 = 1e-12;

/// Apply `BetaCal` to raw `p_up`; returns calibrated P(UP) in [0, 1].
pub fn apply_beta(p_up: f64, cal: &BetaCal) -> f64 {
    let p = p_up.clamp(CLAMP_EPS, 1.0 - CLAMP_EPS);
    let z = cal.a * p.ln() + cal.b * (1.0 - p).ln() + cal.c;
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Fit result: calibration params plus raw val scores for ECE computation.
pub struct BetaCalFit {
    pub cal: BetaCal,
    pub p_up: Vec<f64>,
    pub y: Vec<f64>,
}

/// Fit `(a, b, c)` via 30 Newton steps on the 3×3 Hessian.
/// Returns `IDENTITY` on empty input, single-class input, or divergence.
/// Uses Platt-style label smoothing to keep gradients finite.
pub fn fit_beta_calibration(p_up: &[f64], y: &[f64]) -> BetaCal {
    let n = p_up.len();
    if n == 0 || n != y.len() {
        return IDENTITY;
    }

    let mut log_p: Vec<f64> = Vec::with_capacity(n);
    let mut log_1mp: Vec<f64> = Vec::with_capacity(n);
    let (mut n_pos, mut n_neg) = (0usize, 0usize);
    for i in 0..n {
        let p = p_up[i].clamp(CLAMP_EPS, 1.0 - CLAMP_EPS);
        log_p.push(p.ln());
        log_1mp.push((1.0 - p).ln());
        if y[i] > 0.5 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    if n_pos == 0 || n_neg == 0 {
        return IDENTITY;
    }
    let t_pos = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
    let t_neg = 1.0 / (n_neg as f64 + 2.0);
    let t: Vec<f64> = y.iter().map(|&yi| if yi > 0.5 { t_pos } else { t_neg }).collect();

    let (mut a, mut b, mut c) = (IDENTITY.a, IDENTITY.b, IDENTITY.c);
    for _iter in 0..30 {
        // g = ∑ (σ − t) · x,  H = ∑ w · x xᵀ,  x = [log_p, log_1mp, 1]
        let (mut g_a, mut g_b, mut g_c) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut h_aa, mut h_ab, mut h_ac) = (1e-12_f64, 0.0_f64, 0.0_f64);
        let (mut h_bb, mut h_bc, mut h_cc) = (1e-12_f64, 0.0_f64, 1e-12_f64);
        for i in 0..n {
            let z = a * log_p[i] + b * log_1mp[i] + c;
            let sig = if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            };
            let r = sig - t[i];
            let w = sig * (1.0 - sig);
            let lp = log_p[i];
            let lq = log_1mp[i];
            g_a += r * lp;
            g_b += r * lq;
            g_c += r;
            h_aa += w * lp * lp;
            h_ab += w * lp * lq;
            h_ac += w * lp;
            h_bb += w * lq * lq;
            h_bc += w * lq;
            h_cc += w;
        }

        // Symmetric 3×3 inverse via cofactor expansion.
        let det = h_aa * (h_bb * h_cc - h_bc * h_bc)
                - h_ab * (h_ab * h_cc - h_bc * h_ac)
                + h_ac * (h_ab * h_bc - h_bb * h_ac);
        if !det.is_finite() || det.abs() < 1e-18 {
            return IDENTITY;
        }
        let inv_det = 1.0 / det;
        let c_aa = (h_bb * h_cc - h_bc * h_bc) * inv_det;
        let c_ab = -(h_ab * h_cc - h_bc * h_ac) * inv_det;
        let c_ac = (h_ab * h_bc - h_bb * h_ac) * inv_det;
        let c_bb = (h_aa * h_cc - h_ac * h_ac) * inv_det;
        let c_bc = -(h_aa * h_bc - h_ab * h_ac) * inv_det;
        let c_cc = (h_aa * h_bb - h_ab * h_ab) * inv_det;
        // Newton step: Δθ = H⁻¹ g
        let da = c_aa * g_a + c_ab * g_b + c_ac * g_c;
        let db = c_ab * g_a + c_bb * g_b + c_bc * g_c;
        let dc = c_ac * g_a + c_bc * g_b + c_cc * g_c;
        a -= da;
        b -= db;
        c -= dc;
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return IDENTITY;
        }
        if da.abs() < 1e-9 && db.abs() < 1e-9 && dc.abs() < 1e-9 {
            break;
        }
    }
    let _ = E; // silence unused-import in case E goes unused later
    BetaCal { a, b, c }
}

/// Expected Calibration Error with 10 equal-width bins over [0, 1].
/// Returns 0.0 on empty input.
pub fn expected_calibration_error(p_up: &[f64], y_up: &[f64]) -> f64 {
    if p_up.is_empty() {
        return 0.0;
    }
    const N_BINS: usize = 10;
    let mut bin_sum = [0.0_f64; N_BINS];
    let mut bin_pos = [0.0_f64; N_BINS];
    let mut bin_n = [0usize; N_BINS];
    for (&p, &y) in p_up.iter().zip(y_up.iter()) {
        let p_clamped = p.clamp(0.0, 1.0);
        let mut idx = (p_clamped * N_BINS as f64) as usize;
        if idx >= N_BINS {
            idx = N_BINS - 1;
        }
        bin_sum[idx] += p_clamped;
        bin_pos[idx] += y;
        bin_n[idx] += 1;
    }
    let total = p_up.len() as f64;
    let mut ece = 0.0_f64;
    for i in 0..N_BINS {
        if bin_n[i] == 0 {
            continue;
        }
        let mean_p = bin_sum[i] / bin_n[i] as f64;
        let mean_acc = bin_pos[i] / bin_n[i] as f64;
        ece += (bin_n[i] as f64 / total) * (mean_p - mean_acc).abs();
    }
    ece
}
