// Thin wrapper around the Polymarket CLOB SDK for order placement.

use anyhow::{anyhow, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::str::FromStr;
use tracing::info;

use alloy::signers::Signer as _;
use alloy::signers::local::LocalSigner;
use polymarket_client_sdk_v2::POLYGON;
use polymarket_client_sdk_v2::auth::Normal;
use polymarket_client_sdk_v2::auth::state::Authenticated;
use polymarket_client_sdk_v2::clob::{Client, Config};
use polymarket_client_sdk_v2::clob::types::{AssetType, OrderType, Side, SignatureType};
use polymarket_client_sdk_v2::clob::types::request::BalanceAllowanceRequest;
use polymarket_client_sdk_v2::clob::types::response::OpenOrderResponse;
use polymarket_client_sdk_v2::types::{Address, U256};

/// Returned to the caller after a successful order placement.
pub struct OrderResult {
    pub order_id: String,
    /// Polymarket status: "MATCHED", "LIVE", "CANCELED", "DELAYED", "UNMATCHED", etc.
    pub status: String,
}

/// Snapshot of the funder's USDC.e balance and its exchange allowance,
/// both expressed in human units (USDC, not raw 6-decimal wei).
#[derive(Clone, Debug)]
pub struct BalanceAllowance {
    pub balance: Decimal,
    pub allowance: Decimal,
}

pub struct OrderManager {
    client: Client<Authenticated<Normal>>,
    signer: LocalSigner<alloy::signers::k256::ecdsa::SigningKey>,
    max_usdc: Decimal,
}

impl OrderManager {
    /// Construct and authenticate. `funder` is the Polymarket proxy wallet holding USDC.e.
    pub async fn new(
        api_url: &str,
        private_key: &str,
        funder: Address,
        max_usdc: Decimal,
    ) -> Result<Self> {
        let signer = LocalSigner::from_str(private_key)?.with_chain_id(Some(POLYGON));

        let client = Client::new(api_url, Config::default())?
            .authentication_builder(&signer)
            .signature_type(SignatureType::Proxy)
            .funder(funder)
            .authenticate()
            .await?;

        info!(
            "[OrderManager] authenticated as proxy-funded, funder={} max_usdc={}",
            funder, max_usdc
        );
        Ok(Self { client, signer, max_usdc })
    }

    /// Place a FOK BUY for the UP token. Returns `Ok(None)` if notional exceeds `max_usdc`.
    pub async fn buy_up(
        &self,
        up_token_id: &str,
        price: Decimal,
        shares: Decimal,
    ) -> Result<Option<OrderResult>> {
        self.place_limit_fok(up_token_id, Side::Buy, price, shares).await
    }

    /// Place a FOK BUY for the DOWN token. Returns `Ok(None)` if notional exceeds `max_usdc`.
    pub async fn buy_down(
        &self,
        down_token_id: &str,
        price: Decimal,
        shares: Decimal,
    ) -> Result<Option<OrderResult>> {
        self.place_limit_fok(down_token_id, Side::Buy, price, shares).await
    }

    /// Look up the taker fee rate (basis points) for a token.
    pub async fn fee_rate_bps(&self, token_id: &str) -> Result<u32> {
        let token_id = U256::from_str(token_id)?;
        let resp = self.client.fee_rate_bps(token_id).await?;
        Ok(resp.base_fee)
    }

    /// Re-query a previously placed order's current status.
    pub async fn check_order(&self, order_id: &str) -> Result<OpenOrderResponse> {
        Ok(self.client.order(order_id).await?)
    }

    /// Fetch the funder's USDC.e balance and minimum exchange allowance (scaled to human units).
    pub async fn balance_allowance(&self) -> Result<BalanceAllowance> {
        let req = BalanceAllowanceRequest::builder()
            .asset_type(AssetType::Collateral)
            .build();
        let resp = self.client.balance_allowance(req).await?;

        let scale = dec!(1_000_000);
        let balance = resp.balance / scale;

        if resp.allowances.is_empty() {
            return Err(anyhow!(
                "CLOB /balance-allowance returned no spender allowances for funder"
            ));
        }

        let mut min_allowance: Option<Decimal> = None;
        for (spender, raw) in &resp.allowances {
            let parsed = Decimal::from_str(raw).map_err(|e| {
                anyhow!("allowance for {} is not a decimal string ({}): {}", spender, raw, e)
            })?;
            let human = parsed / scale;
            min_allowance = Some(match min_allowance {
                Some(cur) if cur <= human => cur,
                _ => human,
            });
        }
        let allowance = min_allowance.expect("non-empty map checked above");

        Ok(BalanceAllowance { balance, allowance })
    }

    /// Cancel every open order for this authenticated user.
    pub async fn cancel_all_orders(&self) -> Result<(Vec<String>, usize)> {
        let resp = self.client.cancel_all_orders().await?;
        let not_canceled = resp.not_canceled.len();
        Ok((resp.canceled, not_canceled))
    }

    async fn place_limit_fok(
        &self,
        token_id: &str,
        side: Side,
        price: Decimal,
        shares: Decimal,
    ) -> Result<Option<OrderResult>> {
        let notional = price * shares;
        if notional > self.max_usdc {
            info!(
                "[OrderManager] skip: notional {} exceeds max_usdc {} (price={} shares={})",
                notional, self.max_usdc, price, shares
            );
            return Ok(None);
        }

        let token_id = U256::from_str(token_id)?;

        let order = self
            .client
            .limit_order()
            .token_id(token_id)
            .order_type(OrderType::FOK)
            .price(price)
            .size(shares)
            .side(side)
            .build()
            .await?;

        let signed = self.client.sign(&self.signer, order).await?;
        let response = self.client.post_order(signed).await?;

        let result = OrderResult {
            order_id: response.order_id.clone(),
            status: response.status.to_string(),
        };
        info!(
            "[OrderManager] order placed: id={} status={} notional={}",
            result.order_id, result.status, notional
        );
        Ok(Some(result))
    }
}
