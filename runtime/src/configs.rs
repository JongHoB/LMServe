use clap::{ArgAction, Parser};
use serde::{Deserialize, Serialize};

use crate::types;

macro_rules! default_fn {
    ($name:ident, $ty:ty, $val:expr) => {
        pub fn $name() -> $ty {
            $val
        }
    };
}

default_fn!(
    default_route_policy,
    types::RoutePolicy,
    types::RoutePolicy::RoundRobin
);
default_fn!(default_block_size, usize, 16);
default_fn!(default_gpu_memory_fraction, f32, 0.9);
default_fn!(default_host_kv_cache_size, usize, 16);
default_fn!(default_disk_kv_cache_size, usize, 0);
default_fn!(default_disk_kv_cache_path, String, String::from("/tmp"));
default_fn!(default_enable_reorder, bool, false);
default_fn!(default_max_batch_size, usize, 256);
default_fn!(default_max_seq_len, usize, 16384);
default_fn!(default_max_num_batched_tokens, usize, 2048);
default_fn!(default_tp_size, u8, 1);
default_fn!(
    default_nats_uri,
    String,
    String::from("nats://127.0.0.1:4222")
);

#[derive(Debug, Deserialize, Parser, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ControllerConfig {
    #[arg(long, default_value_t = default_route_policy())]
    #[serde(default = "default_route_policy")]
    pub route_policy: types::RoutePolicy,
}

#[derive(Debug, Deserialize, Parser, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    #[arg(long, default_value_t = default_block_size())]
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    #[arg(long, default_value_t = default_gpu_memory_fraction())]
    #[serde(default = "default_gpu_memory_fraction")]
    pub gpu_memory_fraction: f32,

    #[arg(long, default_value_t = default_host_kv_cache_size())]
    #[serde(default = "default_host_kv_cache_size")]
    pub host_kv_cache_size: usize,

    #[arg(long, default_value_t = default_disk_kv_cache_size())]
    #[serde(default = "default_disk_kv_cache_size")]
    pub disk_kv_cache_size: usize,

    #[arg(long, default_value_t = default_disk_kv_cache_path())]
    #[serde(default = "default_disk_kv_cache_path")]
    pub disk_kv_cache_path: String,

    #[arg(long, action = ArgAction::SetTrue)]
    #[serde(default = "default_enable_reorder")]
    pub enable_reorder: bool,

    #[arg(long, default_value_t = default_max_batch_size())]
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    #[arg(long, default_value_t = default_max_seq_len())]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = default_max_num_batched_tokens())]
    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,

    #[arg(long, default_value_t = default_tp_size())]
    #[serde(default = "default_tp_size")]
    pub tp_size: u8,
}
