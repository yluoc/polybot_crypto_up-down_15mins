// src/feature_engine/normalizer.rs
//
// Per-feature rolling z-score over the last NORM_WINDOW rows, clamped to [-5, 5].
// Stats are kept as (sum, sumsq) for O(1) updates.
// NaN values (per-symbol alignment holes) are excluded from accumulators and
// passed through unchanged; `count[i]` tracks non-NaN entries per feature.

use std::collections::VecDeque;

use super::types::{FeatureRow, TOTAL_FEATURES};

pub const NORM_WINDOW: usize = 200;
pub const NORM_EPS: f64 = 1e-8;

pub struct Normalizer {
    history: [VecDeque<f32>; TOTAL_FEATURES],
    sum:     [f64; TOTAL_FEATURES],
    sumsq:   [f64; TOTAL_FEATURES],
    count:   [usize; TOTAL_FEATURES], // non-NaN entries in history[i]
}

impl Normalizer {
    pub fn new() -> Self {
        Self {
            history: std::array::from_fn(|_| VecDeque::with_capacity(NORM_WINDOW + 1)),
            sum:     [0.0; TOTAL_FEATURES],
            sumsq:   [0.0; TOTAL_FEATURES],
            count:   [0; TOTAL_FEATURES],
        }
    }

    /// Returns None if `raw` is invalid (mirrors C++ early-return). Otherwise
    /// returns a normalised FeatureRow with the same ts_ms + valid=true.
    pub fn push(&mut self, raw: &FeatureRow) -> Option<FeatureRow> {
        if !raw.valid {
            return None;
        }
        let mut out = raw.clone();
        for i in 0..TOTAL_FEATURES {
            let raw_v = raw.features[i];
            let v = raw_v as f64;
            self.history[i].push_back(raw_v);
            // NaN kept in history for window alignment; excluded from accumulators.
            if raw_v.is_nan() {
                // count untouched
            } else {
                self.sum[i]   += v;
                self.sumsq[i] += v * v;
                self.count[i] += 1;
            }
            if self.history[i].len() > NORM_WINDOW {
                if let Some(old) = self.history[i].pop_front() {
                    if !old.is_nan() {
                        let o = old as f64;
                        self.sum[i]   -= o;
                        self.sumsq[i] -= o * o;
                        self.count[i] -= 1;
                    }
                }
            }
            let (mu, sigma) = self.rolling_mean_std(i);
            // NaN passes through; finite → clamped z-score.
            let z = (v - mu) / (sigma + NORM_EPS);
            let z = z.clamp(-5.0, 5.0);
            out.features[i] = z as f32;
        }
        Some(out)
    }

    fn rolling_mean_std(&self, idx: usize) -> (f64, f64) {
        let n = self.count[idx];
        if n == 0 {
            return (0.0, 1.0);
        }
        let mean = self.sum[idx] / n as f64;
        if n < 2 {
            return (mean, 1.0);
        }
        // Guard against FP drift giving a tiny negative variance.
        let var = ((self.sumsq[idx] - self.sum[idx] * mean) / (n - 1) as f64).max(0.0);
        (mean, var.sqrt())
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}
