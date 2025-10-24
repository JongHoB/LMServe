use clap::Parser;
use serde::{Deserialize, Serialize};

use runtime::configs::{ControllerConfig, EngineConfig};
use runtime::types;

use crate::configs;

#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
pub struct CLIArgs {
    #[arg(long)]
    pub config: Option<String>,

    #[arg(long, default_value_t = configs::default_model_name())]
    pub model_name: String,

    #[command(flatten)]
    #[serde(flatten)]
    pub controller: ControllerConfig,

    #[command(flatten)]
    #[serde(flatten)]
    pub engine: EngineConfig,

    #[arg(long, default_value_t = 1)]
    pub num_worker_groups: u32,

    #[arg(long, default_value_t = configs::default_api_address())]
    pub api_address: String,

    #[arg(long, default_value_t = configs::default_nats_uri())]
    pub nats_uri: String,
}

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
pub struct APIServerArgs {
    #[arg(long, default_value_t = configs::default_model_name())]
    pub model_name: String,

    #[command(flatten)]
    #[serde(flatten)]
    pub controller: ControllerConfig,

    #[arg(long, default_value_t = configs::default_api_address())]
    pub api_address: String,

    #[arg(
        long,
        value_delimiter = ' ',
        num_args = 1..,
        default_values_t = vec![configs::default_srv_address()]
    )]
    pub llm_server_addresses: Vec<String>,

    #[arg(long, default_value_t = configs::default_nats_uri())]
    pub nats_uri: String,
}

#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
pub struct LLMSrvArgs {
    #[arg(long, default_value = "all")]
    pub kind: types::EngineKind,

    #[arg(long, default_value_t = configs::default_model_name())]
    pub model_name: String,

    #[command(flatten)]
    #[serde(flatten)]
    pub engine: EngineConfig,

    #[arg(long, default_value_t = configs::default_srv_address())]
    pub address: String,

    #[arg(long, value_delimiter = ' ')]
    pub devices: Option<Vec<u8>>,

    #[arg(long, default_value_t = configs::default_srv_address())]
    pub nats_uri: String,
}
