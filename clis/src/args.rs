use clap::Parser;
use serde::{Deserialize, Serialize};

use crate::configs;

#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
pub struct CLIArgs {
    #[arg(long)]
    pub config: Option<String>,

    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    pub model_name: String,

    #[arg(long, default_value_t = configs::default_block_size())]
    pub block_size: usize,

    /// Fraction of total GPU memory to use (0.0 ~ 1.0)
    #[arg(long, default_value_t = configs::default_gpu_memory_fraction())]
    pub gpu_memory_fraction: f32,

    // Host-side KV cache size in GB
    #[arg(long, default_value_t = configs::default_host_kv_cache_size())]
    pub host_kv_cache_size: usize,

    #[arg(long, default_value_t = configs::default_max_batch_size())]
    pub max_batch_size: usize,

    #[arg(long, default_value_t = configs::default_max_seq_len())]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = configs::default_max_num_batched_tokens())]
    pub max_num_batched_tokens: usize,

    #[arg(long, default_value_t = configs::default_tp_size())]
    pub tp_size: u8,

    #[arg(long, default_value_t = 1)]
    pub num_worker_groups: u32,

    #[arg(long, default_value = "127.0.0.1:8000")]
    pub address: String,
}

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

#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
pub struct LLMSrvArgs {
    #[arg(long, default_value = "all")]
    pub kind: String,

    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    pub model_name: String,

    #[arg(long, default_value_t = configs::default_block_size())]
    pub block_size: usize,

    /// Fraction of total GPU memory to use (0.0 ~ 1.0)
    #[arg(long, default_value_t = configs::default_gpu_memory_fraction())]
    pub gpu_memory_fraction: f32,

    // Host-side KV cache size in GB
    #[arg(long, default_value_t = configs::default_host_kv_cache_size())]
    pub host_kv_cache_size: usize,

    #[arg(long, default_value_t = configs::default_max_batch_size())]
    pub max_batch_size: usize,

    #[arg(long, default_value_t = configs::default_max_seq_len())]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = configs::default_max_num_batched_tokens())]
    pub max_num_batched_tokens: usize,

    #[arg(long, default_value_t = configs::default_tp_size())]
    pub tp_size: u8,

    #[arg(long, default_value = "127.0.0.1:7000")]
    pub address: String,

    #[arg(long, value_delimiter = ' ')]
    pub devices: Option<Vec<u8>>,
}
