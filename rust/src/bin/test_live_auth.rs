//! Standalone Polymarket authentication and connectivity check.
//!
//! Usage: `cargo run --bin test_live_auth`
//!
//! Reads POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER, POLYMARKET_API_URL and
//! performs three probes: GET /ok, GET /time, and /auth/api-key (L1-signed).
//! Exits 0 on success, non-zero on any failure.

use std::process::ExitCode;
use std::str::FromStr;

use alloy::signers::Signer as _;
use alloy::signers::local::LocalSigner;
use anyhow::{Context, Result};
use polymarket_client_sdk_v2::POLYGON;
use polymarket_client_sdk_v2::clob::{Client, Config};
use polymarket_client_sdk_v2::clob::types::SignatureType;
use polymarket_client_sdk_v2::types::Address;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => {
            println!("\nAUTH OK — ready for live trading");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("FAIL: {err:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    dotenvy::dotenv().ok();

    let api_url = std::env::var("POLYMARKET_API_URL")
        .unwrap_or_else(|_| "https://clob.polymarket.com".into());
    let private_key = std::env::var("POLYMARKET_PRIVATE_KEY")
        .context("POLYMARKET_PRIVATE_KEY is not set")?;
    let funder_raw = std::env::var("POLYMARKET_FUNDER")
        .context("POLYMARKET_FUNDER is not set")?;
    let funder = Address::from_str(&funder_raw)
        .context("POLYMARKET_FUNDER must be a 0x-prefixed checksum address")?;

    println!("host={api_url}");
    println!("funder={funder}");

    let signer = LocalSigner::from_str(&private_key)
        .context("POLYMARKET_PRIVATE_KEY is not a valid hex signing key")?
        .with_chain_id(Some(POLYGON));
    println!("signer_eoa={}", signer.address());

    let client = Client::new(&api_url, Config::default())
        .context("failed to construct unauthenticated CLOB client")?;

    // 1. GET /ok
    let ok = client.ok().await.context("GET /ok failed")?;
    println!("ok(): {ok}");

    // 2. GET /time
    let server_time = client
        .server_time()
        .await
        .context("GET /time failed")?;
    println!("server_time(): {server_time:?}");

    // 3. L1-signed /auth/api-key
    let authed = client
        .authentication_builder(&signer)
        .signature_type(SignatureType::Proxy)
        .funder(funder)
        .authenticate()
        .await
        .context("authenticate() failed — L1 signature or funder address rejected")?;

    println!(
        "create_or_derive_api_creds(): OK (api_key={})",
        authed.credentials().key()
    );

    Ok(())
}
