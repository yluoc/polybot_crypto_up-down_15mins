// src/inference/predict.rs — argmax helper for 2-class predict output.
// Class 0 = UP, class 1 = DOWN.

use crate::calibration::{apply_beta, BetaCal};
use crate::signal::Action;

#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    pub action: Action,
    /// Probability of the chosen class in [0, 1]. Used as the confidence gate.
    pub confidence: f32,
    /// |P(UP) - P(DOWN)|; kept for log-parity with historical signals.
    pub raw_score: f32,
}

/// Argmax on 2-class output after beta calibration. Panics if `probs.len() < 2`.
pub fn predict_one(probs: &[f64], calibration: BetaCal) -> Prediction {
    assert!(probs.len() >= 2, "expected 2 class probs, got {}", probs.len());

    let up = apply_beta(probs[0], &calibration);
    let down = 1.0 - up;

    // Tie breaks toward UP.
    let (action, confidence) = if down > up {
        (Action::BuyDown, down)
    } else {
        (Action::BuyUp, up)
    };

    let margin = (up - down).abs();

    Prediction {
        action,
        confidence: confidence as f32,
        raw_score: margin as f32,
    }
}
