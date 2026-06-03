// Startup wallet pre-flight check: verifies funder USDC.e balance and
// exchange allowance before the main loop begins. Bypassable via
// PREFLIGHT_ENABLED=false for dry-run workflows.

use anyhow::{bail, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use sqlx::PgPool;
use tracing::{info, warn};

use crate::config::Config;
use crate::db::queries;
use crate::order_manager::{BalanceAllowance, OrderManager};

/// Outcome of evaluating a `BalanceAllowance` against `order_max_usdc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightResult {
    /// Allowance and balance both cover at least 4× one max order.
    Ok,
    /// Allowance covers one max order, but balance is under 4× — warn only.
    Warn,
    /// Balance cannot cover a single max order — bail.
    FatalBalance,
    /// Allowance cannot cover a single max order — bail (operator must run
    /// the SDK `approvals` example).
    FatalAllowance,
}

/// Pure threshold function — no I/O. Allowance is checked before balance.
pub fn evaluate(ba: &BalanceAllowance, order_max_usdc: Decimal) -> PreflightResult {
    if ba.allowance < order_max_usdc {
        return PreflightResult::FatalAllowance;
    }
    if ba.balance < order_max_usdc {
        return PreflightResult::FatalBalance;
    }
    if ba.balance < order_max_usdc * dec!(4) {
        return PreflightResult::Warn;
    }
    PreflightResult::Ok
}

/// Run the pre-flight wallet check. No-op when `cfg.preflight_enabled` is false.
pub async fn run(orders: &OrderManager, cfg: &Config, pool: &PgPool) -> Result<()> {
    if !cfg.preflight_enabled {
        warn!("[preflight] PREFLIGHT_ENABLED=false — skipping wallet check");
        return Ok(());
    }

    let ba = orders.balance_allowance().await?;
    let decision = evaluate(&ba, cfg.order_max_usdc);
    match decision {
        PreflightResult::Ok => {
            info!(
                "[preflight] OK: balance={} allowance={} funder={}",
                ba.balance, ba.allowance, cfg.funder
            );
            Ok(())
        }
        PreflightResult::Warn => {
            let detail = format!(
                "balance {} < 4×max_order ({}); funder={}",
                ba.balance,
                cfg.order_max_usdc * dec!(4),
                cfg.funder
            );
            warn!("[preflight] low balance: {}", detail);
            if let Err(e) = queries::insert_error(pool, "rust", "preflight_low_balance", &detail).await {
                warn!("[preflight] could not persist low-balance note: {:#}", e);
            }
            Ok(())
        }
        PreflightResult::FatalAllowance => {
            bail!(
                "preflight: allowance {} < max order {}; run `cargo run --example approvals` \
                 from the polymarket_client_sdk_v2 to grant the exchange contracts spending rights",
                ba.allowance,
                cfg.order_max_usdc
            );
        }
        PreflightResult::FatalBalance => {
            bail!(
                "preflight: funder {} balance {} cannot cover one order of {}",
                cfg.funder,
                ba.balance,
                cfg.order_max_usdc
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ba(balance: Decimal, allowance: Decimal) -> BalanceAllowance {
        BalanceAllowance { balance, allowance }
    }

    #[test]
    fn ok_when_balance_well_above_4x_and_allowance_ample() {
        let result = evaluate(&ba(dec!(100), dec!(1_000_000)), dec!(10));
        assert_eq!(result, PreflightResult::Ok);
    }

    #[test]
    fn warn_when_balance_covers_one_order_but_under_4x() {
        let result = evaluate(&ba(dec!(25), dec!(1_000_000)), dec!(10));
        assert_eq!(result, PreflightResult::Warn);
    }

    #[test]
    fn fatal_balance_when_balance_below_one_max_order() {
        let result = evaluate(&ba(dec!(5), dec!(1_000_000)), dec!(10));
        assert_eq!(result, PreflightResult::FatalBalance);
    }

    #[test]
    fn fatal_allowance_when_allowance_below_one_max_order() {
        let result = evaluate(&ba(dec!(10_000), dec!(0)), dec!(10));
        assert_eq!(result, PreflightResult::FatalAllowance);
    }

    #[test]
    fn fatal_allowance_takes_precedence_over_fatal_balance() {
        let result = evaluate(&ba(dec!(0), dec!(0)), dec!(10));
        assert_eq!(result, PreflightResult::FatalAllowance);
    }

    #[test]
    fn boundary_balance_exactly_4x_is_ok() {
        let result = evaluate(&ba(dec!(40), dec!(1_000_000)), dec!(10));
        assert_eq!(result, PreflightResult::Ok);
    }

    #[test]
    fn boundary_allowance_exactly_max_order_is_ok() {
        let result = evaluate(&ba(dec!(40), dec!(10)), dec!(10));
        assert_eq!(result, PreflightResult::Ok);
    }
}
