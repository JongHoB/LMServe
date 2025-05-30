use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
pub struct APIServerArgs {
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    pub model_name: String,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub address: String,

    #[arg(
        long,
        value_delimiter = ' ',
        num_args = 1..,
        default_values_t = vec!["127.0.0.1:7000".to_string()]
    )]
    pub llm_server_addresses: Vec<String>,
}
