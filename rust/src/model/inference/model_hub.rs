// src/inference/model_hub.rs
//
// Per-symbol LightGBM Booster hub. Models are 2-class (UP/DOWN).
// Hot-reload swaps one symbol's Booster under exclusive write lock.
// Blobs with an unsupported format_version are skipped with a warning.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use lightgbm3::{Booster, ImportanceType};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::calibration::{self, BetaCal, BETA_MAGIC, BETA_POOLED_MAGIC, BETA_POOLED_TRAILER_LEN, BETA_TRAILER_LEN};
use crate::feature_engine::{INSTRUMENT_COUNT, TOTAL_FEATURES};
use crate::signal::{symbol_id, Symbol};
use crate::training::lr_l1::LrL1Model;

/// Current model blob format version. Older blobs outside
/// `[MIN_SUPPORTED_FORMAT_VERSION, CURRENT_FORMAT_VERSION]` are skipped at load.
pub const CURRENT_FORMAT_VERSION: i16 = 16;

/// Minimum format_version the hub will load. Exists to support the pooled
/// rollback path; raise to `CURRENT_FORMAT_VERSION` once no v14 blobs remain.
pub const MIN_SUPPORTED_FORMAT_VERSION: i16 = 14;

/// Input dimension of the pooled LightGBM booster: `TOTAL_FEATURES` + 1
/// categorical `symbol_id` slot injected by `PooledAdapter` at predict time.
pub const POOLED_INPUT_DIM: usize = TOTAL_FEATURES + 1;

// Legacy v14/v15 Platt trailer markers; v16 uses BETA_MAGIC / BETA_POOLED_MAGIC.
// v15 blobs are converted to BetaCal at load via `calibration::platt_to_beta`.
const PLATT_MAGIC: &[u8; 4] = b"PLAT";
const PLATT_TRAILER_LEN: usize = 8 + 8 + 4;
const PLATT_POOLED_MAGIC: &[u8; 4] = b"PLPL";
const PLATT_POOLED_TRAILER_LEN: usize = 8 * 2 * INSTRUMENT_COUNT + 4;

/// Parse the per-symbol calibration trailer; dispatch on `format_version`.
/// v16+: `BETA_MAGIC` + 3 f64s. v14/v15: `PLATT_MAGIC` + 2 f64s, converted via
/// `calibration::platt_to_beta`. Missing/mismatched trailer falls back to `IDENTITY`.
fn parse_per_symbol_trailer(bytes: &[u8], format_version: i16) -> Result<(&[u8], BetaCal)> {
    if format_version >= 16 {
        if bytes.len() >= BETA_TRAILER_LEN && &bytes[bytes.len() - 4..] == BETA_MAGIC {
            let split = bytes.len() - BETA_TRAILER_LEN;
            let a = f64::from_le_bytes(bytes[split..split + 8].try_into().unwrap());
            let b = f64::from_le_bytes(bytes[split + 8..split + 16].try_into().unwrap());
            let c = f64::from_le_bytes(bytes[split + 16..split + 24].try_into().unwrap());
            Ok((&bytes[..split], BetaCal { a, b, c }))
        } else {
            warn!(
                "v{} blob missing BETA trailer (got tail {:?}); falling back to IDENTITY",
                format_version,
                bytes.get(bytes.len().saturating_sub(4)..).unwrap_or(&[])
            );
            Ok((bytes, calibration::IDENTITY))
        }
    } else if bytes.len() >= PLATT_TRAILER_LEN && &bytes[bytes.len() - 4..] == PLATT_MAGIC {
        let split = bytes.len() - PLATT_TRAILER_LEN;
        let a = f64::from_le_bytes(bytes[split..split + 8].try_into().unwrap());
        let b = f64::from_le_bytes(bytes[split + 8..split + 16].try_into().unwrap());
        Ok((&bytes[..split], calibration::platt_to_beta(a, b)))
    } else {
        // No recognised trailer — fall back to IDENTITY.
        Ok((bytes, calibration::IDENTITY))
    }
}

/// Per-symbol LightGBM booster wrapped for thread-safety.
pub struct SafeBooster {
    booster: Booster,
    /// Beta calibration applied to P(UP). `calibration::IDENTITY` is the no-op.
    calibration: BetaCal,
    /// `model_versions.id` of the loaded blob.
    model_version_id: i64,
    /// Gain importance vector, length `TOTAL_FEATURES`, cached at load time.
    feature_importances: Vec<f64>,
}

// SAFETY: LightGBM prediction is thread-safe on a shared Booster;
// all mutation goes through RwLock::write.
unsafe impl Send for SafeBooster {}
unsafe impl Sync for SafeBooster {}

impl SafeBooster {
    /// Parse a model blob into a `SafeBooster`. Trailer dispatch on `format_version`:
    /// v16 → `BETA` + 3 f64s; v14/v15 → `PLAT` + 2 f64s converted via
    /// `calibration::platt_to_beta`; no trailer → `calibration::IDENTITY`.
    pub fn from_bytes(
        bytes: &[u8],
        model_version_id: i64,
        format_version: i16,
    ) -> Result<Self> {
        let (text_bytes, calibration) = parse_per_symbol_trailer(bytes, format_version)?;
        let text = std::str::from_utf8(text_bytes)
            .context("model blob is not valid UTF-8 — LightGBM save_string produces text")?;
        let booster = Booster::from_string(text)
            .map_err(|e| anyhow!("Booster::from_string: {e}"))?;
        let n_classes = booster.num_classes();
        if n_classes != 2 {
            return Err(anyhow!(
                "expected 2-class LightGBM model (UP/DOWN), got num_classes={n_classes}"
            ));
        }
        let n_features = booster.num_features();
        if n_features as usize != TOTAL_FEATURES {
            return Err(anyhow!(
                "expected num_features={TOTAL_FEATURES}, got {n_features}"
            ));
        }
        let feature_importances = booster
            .feature_importance(ImportanceType::Gain)
            .map_err(|e| anyhow!("Booster::feature_importance: {e}"))?;
        if feature_importances.len() != TOTAL_FEATURES {
            return Err(anyhow!(
                "expected feature_importance len={TOTAL_FEATURES}, got {}",
                feature_importances.len()
            ));
        }
        Ok(SafeBooster {
            booster,
            calibration,
            model_version_id,
            feature_importances,
        })
    }

    /// Predict `[P(UP), P(DOWN)]` for one feature row.
    pub fn predict_one(&self, features: &[f32; TOTAL_FEATURES]) -> Result<[f64; 2]> {
        let out = self
            .booster
            .predict::<f32>(features.as_slice(), TOTAL_FEATURES as i32, true)
            .map_err(|e| anyhow!("Booster::predict: {e}"))?;
        if out.len() != 2 {
            return Err(anyhow!(
                "LightGBM returned {} outputs, expected 2 (num_class)",
                out.len()
            ));
        }
        Ok([out[0], out[1]])
    }

    pub fn calibration(&self) -> BetaCal {
        self.calibration
    }

    pub fn model_version_id(&self) -> i64 {
        self.model_version_id
    }

    /// Cached gain importance vector, length `TOTAL_FEATURES`.
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }
}

/// Pooled multi-task LightGBM booster shared across all 4 symbol slots via `Arc`.
/// Expects a `POOLED_INPUT_DIM`-wide row; per-symbol calibration is on `PooledAdapter`.
pub struct PooledBooster {
    booster: Booster,
    /// Gain importance vector, length `POOLED_INPUT_DIM`.
    feature_importances: Vec<f64>,
}

// SAFETY: LightGBM prediction is thread-safe on a shared Booster;
// all mutation is via RwLock::write.
unsafe impl Send for PooledBooster {}
unsafe impl Sync for PooledBooster {}

impl PooledBooster {
    /// Parse a pooled blob into `(PooledBooster, [BetaCal; INSTRUMENT_COUNT])`.
    /// v16+: `BETA_POOLED_MAGIC` + 4×3 f64s; v15: `PLATT_POOLED_MAGIC` + 4×2 f64s
    /// converted via `calibration::platt_to_beta`. Missing trailer is a hard error.
    pub fn from_bytes(
        bytes: &[u8],
        format_version: i16,
    ) -> Result<(Self, [BetaCal; INSTRUMENT_COUNT])> {
        let (text_bytes, betas) = if format_version >= 16 {
            if bytes.len() < BETA_POOLED_TRAILER_LEN
                || &bytes[bytes.len() - 4..] != BETA_POOLED_MAGIC
            {
                return Err(anyhow!(
                    "v{format_version} pooled blob missing BETA_POOLED trailer \
                     (expected {INSTRUMENT_COUNT}-BetaCal calibration)"
                ));
            }
            let split = bytes.len() - BETA_POOLED_TRAILER_LEN;
            let mut betas = [calibration::IDENTITY; INSTRUMENT_COUNT];
            for i in 0..INSTRUMENT_COUNT {
                let off = split + i * 24;
                let a = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                let b = f64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
                let c = f64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap());
                betas[i] = BetaCal { a, b, c };
            }
            (&bytes[..split], betas)
        } else {
            if bytes.len() < PLATT_POOLED_TRAILER_LEN
                || &bytes[bytes.len() - 4..] != PLATT_POOLED_MAGIC
            {
                return Err(anyhow!(
                    "v{format_version} pooled blob missing PLATT_POOLED trailer \
                     (expected {INSTRUMENT_COUNT}-Platt calibration)"
                ));
            }
            let split = bytes.len() - PLATT_POOLED_TRAILER_LEN;
            let mut betas = [calibration::IDENTITY; INSTRUMENT_COUNT];
            for i in 0..INSTRUMENT_COUNT {
                let off = split + i * 16;
                let a = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                let b = f64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
                betas[i] = calibration::platt_to_beta(a, b);
            }
            (&bytes[..split], betas)
        };
        let text = std::str::from_utf8(text_bytes)
            .context("pooled model blob is not valid UTF-8 — LightGBM save_string produces text")?;
        let booster = Booster::from_string(text)
            .map_err(|e| anyhow!("Booster::from_string (pooled): {e}"))?;
        let n_classes = booster.num_classes();
        if n_classes != 2 {
            return Err(anyhow!(
                "expected 2-class pooled LightGBM model, got num_classes={n_classes}"
            ));
        }
        let n_features = booster.num_features();
        if n_features as usize != POOLED_INPUT_DIM {
            return Err(anyhow!(
                "expected pooled num_features={POOLED_INPUT_DIM}, got {n_features}"
            ));
        }
        let feature_importances = booster
            .feature_importance(ImportanceType::Gain)
            .map_err(|e| anyhow!("Booster::feature_importance (pooled): {e}"))?;
        if feature_importances.len() != POOLED_INPUT_DIM {
            return Err(anyhow!(
                "expected pooled feature_importance len={POOLED_INPUT_DIM}, got {}",
                feature_importances.len()
            ));
        }
        Ok((
            PooledBooster {
                booster,
                feature_importances,
            },
            betas,
        ))
    }

    /// Predict `[P(UP), P(DOWN)]` for one `POOLED_INPUT_DIM`-wide row.
    fn predict_one_pooled(&self, features: &[f32; POOLED_INPUT_DIM]) -> Result<[f64; 2]> {
        let out = self
            .booster
            .predict::<f32>(features.as_slice(), POOLED_INPUT_DIM as i32, true)
            .map_err(|e| anyhow!("Booster::predict (pooled): {e}"))?;
        if out.len() != 2 {
            return Err(anyhow!(
                "LightGBM (pooled) returned {} outputs, expected 2",
                out.len()
            ));
        }
        Ok([out[0], out[1]])
    }

    /// Cached gain importance vector, length `POOLED_INPUT_DIM`.
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }
}

/// Per-symbol adapter over a shared `PooledBooster`. Each adapter pins its
/// own `symbol_id_f32` (injected at slot `TOTAL_FEATURES` on predict) and
/// its own per-symbol `BetaCal`.
pub struct PooledAdapter {
    booster: Arc<PooledBooster>,
    /// Categorical symbol_id as f32, written at slot `TOTAL_FEATURES` on every predict.
    symbol_id_f32: f32,
    beta: BetaCal,
    /// `model_versions.id` of this symbol's row (distinct per symbol even for pooled blobs).
    model_version_id: i64,
}

impl PooledAdapter {
    pub fn new(
        booster: Arc<PooledBooster>,
        betas: [BetaCal; INSTRUMENT_COUNT],
        sym: Symbol,
        model_version_id: i64,
    ) -> Self {
        let sid = symbol_id(sym);
        PooledAdapter {
            booster,
            symbol_id_f32: sid as f32,
            beta: betas[sid as usize],
            model_version_id,
        }
    }

    fn predict_up_down(&self, features: &[f32; TOTAL_FEATURES]) -> Result<[f64; 2]> {
        let mut buf = [0.0_f32; POOLED_INPUT_DIM];
        buf[..TOTAL_FEATURES].copy_from_slice(features);
        buf[TOTAL_FEATURES] = self.symbol_id_f32;
        self.booster.predict_one_pooled(&buf)
    }

    pub fn calibration(&self) -> BetaCal {
        self.beta
    }

    pub fn model_version_id(&self) -> i64 {
        self.model_version_id
    }

    /// Gain importance for the first `TOTAL_FEATURES` slots (symbol_id slot suppressed).
    pub fn feature_importances(&self) -> &[f64] {
        &self.booster.feature_importances()[..TOTAL_FEATURES]
    }

    /// Full `POOLED_INPUT_DIM`-wide gain importance, including the symbol_id slot. Diagnostics only.
    #[allow(dead_code)]
    pub fn feature_importances_full(&self) -> &[f64] {
        self.booster.feature_importances()
    }
}

/// One symbol's promoted model: LightGBM, LR-L1 baseline, or pooled LightGBM.
pub enum SymbolModel {
    LightGbm(SafeBooster),
    LrL1 {
        model: LrL1Model,
        model_version_id: i64,
        /// `|standardized coefficient|` per feature, length `TOTAL_FEATURES`.
        importances: Vec<f64>,
    },
    /// Pooled multi-task LightGBM — shared `Arc<PooledBooster>` deduped by sha256.
    LightGbmPooled(PooledAdapter),
}

impl SymbolModel {
    /// Deserialize a blob into the variant named by `model_family`.
    /// `'lightgbm_pooled'` at hot-reload time builds a non-deduplicated adapter
    /// (dedup is startup-only via `from_bytes_map`).
    pub fn from_bytes(
        model_family: &str,
        bytes: &[u8],
        model_version_id: i64,
        sym: Symbol,
        format_version: i16,
    ) -> Result<Self> {
        match model_family {
            "lightgbm" => Ok(SymbolModel::LightGbm(SafeBooster::from_bytes(
                bytes,
                model_version_id,
                format_version,
            )?)),
            "lr_l1" => {
                let model = LrL1Model::from_bytes(bytes, format_version)?;
                let importances = model.abs_coefficients();
                Ok(SymbolModel::LrL1 {
                    model,
                    model_version_id,
                    importances,
                })
            }
            "lightgbm_pooled" => {
                let (booster, betas) = PooledBooster::from_bytes(bytes, format_version)?;
                Ok(SymbolModel::LightGbmPooled(PooledAdapter::new(
                    Arc::new(booster),
                    betas,
                    sym,
                    model_version_id,
                )))
            }
            other => Err(anyhow!("unknown model_family '{other}'")),
        }
    }

    /// Predict `[P(UP), P(DOWN)]` plus `BetaCal` for one feature row.
    pub fn predict_up_down(
        &self,
        features: &[f32; TOTAL_FEATURES],
    ) -> Result<([f64; 2], BetaCal)> {
        match self {
            SymbolModel::LightGbm(b) => Ok((b.predict_one(features)?, b.calibration())),
            SymbolModel::LrL1 { model, .. } => {
                let p_up = model.predict_up(features);
                Ok(([p_up, 1.0 - p_up], model.beta()))
            }
            SymbolModel::LightGbmPooled(adapter) => {
                Ok((adapter.predict_up_down(features)?, adapter.calibration()))
            }
        }
    }

    /// Feature-importance vector, length `TOTAL_FEATURES` for all families.
    pub fn feature_importances(&self) -> &[f64] {
        match self {
            SymbolModel::LightGbm(b) => b.feature_importances(),
            SymbolModel::LrL1 { importances, .. } => importances,
            SymbolModel::LightGbmPooled(adapter) => adapter.feature_importances(),
        }
    }

    pub fn model_version_id(&self) -> i64 {
        match self {
            SymbolModel::LightGbm(b) => b.model_version_id(),
            SymbolModel::LrL1 {
                model_version_id, ..
            } => *model_version_id,
            SymbolModel::LightGbmPooled(adapter) => adapter.model_version_id(),
        }
    }
}

/// One row of `db::queries::load_current_models` output.
#[derive(Debug, Clone)]
pub struct ModelBlobEntry {
    pub bytes: Vec<u8>,
    pub format_version: i16,
    pub model_version_id: i64,
    pub model_family: String,
    /// SHA-256 hex of `model_bytes`; used to dedup pooled blobs across symbol slots.
    pub sha256_hex: String,
}

pub struct ModelHub {
    models: HashMap<Symbol, Arc<RwLock<SymbolModel>>>,
}

impl ModelHub {
    /// Build a `ModelHub` from a `symbol_short → ModelBlobEntry` map.
    /// Unsupported format_versions are skipped with a warning. Pooled blobs
    /// with matching `sha256_hex` share one `Arc<PooledBooster>`.
    pub fn from_bytes_map(map: HashMap<String, ModelBlobEntry>) -> Result<Self> {
        let mut models = HashMap::with_capacity(map.len());
        let mut loaded = 0usize;
        let mut skipped_legacy = 0usize;
        let mut pooled_dedup_groups = 0usize;
        let mut pooled_dedup_fallbacks = 0usize;

        struct Parsed {
            sym: Symbol,
            entry: ModelBlobEntry,
            short: String,
        }
        let mut parsed: Vec<Parsed> = Vec::with_capacity(map.len());
        for (short, entry) in map {
            let sym = match Symbol::from_str_ci(&short) {
                Ok(s) => s,
                Err(e) => {
                    warn!("[ModelHub] skipping unknown symbol '{short}': {e}");
                    continue;
                }
            };
            if !(MIN_SUPPORTED_FORMAT_VERSION..=CURRENT_FORMAT_VERSION).contains(&entry.format_version) {
                warn!(
                    "[ModelHub] skipping out-of-range format_version={} model for {} — \
                     supported range is [{}, {}] (FEATURE_DIM={}, TOTAL_FEATURES={}); \
                     bootstrap will re-train",
                    entry.format_version, short,
                    MIN_SUPPORTED_FORMAT_VERSION, CURRENT_FORMAT_VERSION,
                    crate::feature_engine::FEATURE_DIM, TOTAL_FEATURES
                );
                skipped_legacy += 1;
                continue;
            }
            parsed.push(Parsed { sym, entry, short });
        }

        let mut pooled_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, p) in parsed.iter().enumerate() {
            if p.entry.model_family == "lightgbm_pooled" {
                pooled_groups
                    .entry(p.entry.sha256_hex.clone())
                    .or_default()
                    .push(i);
            }
        }
        let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (sha, idxs) in &pooled_groups {
            if idxs.len() < 2 {
                continue;
            }
            let head = &parsed[idxs[0]].entry;
            let consistent = idxs.iter().all(|&j| {
                let e = &parsed[j].entry;
                e.format_version == head.format_version && e.model_family == head.model_family
            });
            if !consistent {
                warn!(
                    "[ModelHub] pooled sha256={} group has inconsistent metadata \
                     across {} symbols (format_version or model_family mismatch); \
                     refusing to dedup — each entry will load independently",
                    sha, idxs.len()
                );
                pooled_dedup_fallbacks += 1;
                continue;
            }
            let (booster, betas) = PooledBooster::from_bytes(&head.bytes, head.format_version)
                .with_context(|| format!("pooled blob load (sha256={sha})"))?;
            let booster = Arc::new(booster);
            for &j in idxs {
                let p = &parsed[j];
                let adapter = PooledAdapter::new(
                    Arc::clone(&booster),
                    betas,
                    p.sym,
                    p.entry.model_version_id,
                );
                models.insert(
                    p.sym,
                    Arc::new(RwLock::new(SymbolModel::LightGbmPooled(adapter))),
                );
                info!(
                    "[ModelHub] loaded pooled model for {} (sha256={}, shared Arc)",
                    p.short, sha
                );
                loaded += 1;
                consumed.insert(j);
            }
            pooled_dedup_groups += 1;
        }

        for (i, p) in parsed.iter().enumerate() {
            if consumed.contains(&i) {
                continue;
            }
            let model = SymbolModel::from_bytes(
                &p.entry.model_family,
                &p.entry.bytes,
                p.entry.model_version_id,
                p.sym,
                p.entry.format_version,
            )
            .with_context(|| format!("load model for {}", p.short))?;
            models.insert(p.sym, Arc::new(RwLock::new(model)));
            info!(
                "[ModelHub] loaded model for {} family={} ({} bytes)",
                p.short,
                p.entry.model_family,
                p.entry.bytes.len()
            );
            loaded += 1;
        }

        info!(
            "[ModelHub] startup summary — loaded={} skipped_legacy_format={} \
             pooled_dedup_groups={} pooled_dedup_fallbacks={} \
             (format_version range=[{}, {}])",
            loaded, skipped_legacy, pooled_dedup_groups, pooled_dedup_fallbacks,
            MIN_SUPPORTED_FORMAT_VERSION, CURRENT_FORMAT_VERSION
        );
        Ok(Self { models })
    }

    pub fn get(&self, sym: Symbol) -> Option<Arc<RwLock<SymbolModel>>> {
        self.models.get(&sym).cloned()
    }

    pub fn has(&self, sym: Symbol) -> bool {
        self.models.contains_key(&sym)
    }

    /// Hot-swap the model for `sym`. Deserialization runs outside the write lock.
    /// Blobs with an unsupported `format_version` are rejected with a warning.
    pub async fn reload(
        &mut self,
        sym: Symbol,
        bytes: &[u8],
        format_version: i16,
        model_version_id: i64,
        model_family: &str,
    ) -> Result<()> {
        if !(MIN_SUPPORTED_FORMAT_VERSION..=CURRENT_FORMAT_VERSION).contains(&format_version) {
            warn!(
                "[ModelHub] reload({}) skipped: format_version={} outside supported range [{}, {}]",
                sym.as_str(),
                format_version,
                MIN_SUPPORTED_FORMAT_VERSION,
                CURRENT_FORMAT_VERSION
            );
            return Ok(());
        }
        let new_model = SymbolModel::from_bytes(
            model_family,
            bytes,
            model_version_id,
            sym,
            format_version,
        )
        .with_context(|| format!("reload model for {:?}", sym))?;
        match self.models.get(&sym) {
            Some(slot) => {
                let mut guard = slot.write().await;
                *guard = new_model;
                info!(
                    "[ModelHub] reloaded {} family={} ({} bytes)",
                    sym.as_str(),
                    model_family,
                    bytes.len()
                );
            }
            None => {
                self.models.insert(sym, Arc::new(RwLock::new(new_model)));
                info!(
                    "[ModelHub] first load for {} family={} ({} bytes)",
                    sym.as_str(),
                    model_family,
                    bytes.len()
                );
            }
        }
        Ok(())
    }
}
