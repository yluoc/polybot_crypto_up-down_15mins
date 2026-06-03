pub mod model_hub;
pub mod predict;

pub use model_hub::ModelHub;
pub use predict::{Prediction, predict_one};
