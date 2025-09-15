use runtime::types;
use serde::Deserialize;

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
default_fn!(default_max_batch_size, usize, 256);
default_fn!(default_max_seq_len, usize, 16384);
default_fn!(default_max_num_batched_tokens, usize, 512);
default_fn!(default_tp_size, u8, 1);
default_fn!(
    default_nats_uri,
    String,
    String::from("nats://127.0.0.1:4222")
);

#[derive(Debug, Deserialize)]
pub struct LLMCluConfig {
    pub model_name: String,

    pub api_server: APIServerConfig,

    pub controller: ControllerConfig,

    pub llm_servers: Vec<LLMSrvConfig>,
}

#[derive(Debug, Deserialize)]
pub struct APIServerConfig {
    pub address: String,
}

#[derive(Debug, Deserialize)]
pub struct ControllerConfig {
    #[serde(default = "default_route_policy")]
    pub route_policy: types::RoutePolicy,

    #[serde(default = "default_nats_uri")]
    pub nats_uri: String,
}

#[derive(Debug, Deserialize)]
pub struct LLMSrvConfig {
    pub kind: types::EngineKind,

    pub address: String,

    #[serde(default = "default_block_size")]
    pub block_size: usize,

    #[serde(default = "default_gpu_memory_fraction")]
    pub gpu_memory_fraction: f32,

    #[serde(default = "default_host_kv_cache_size")]
    pub host_kv_cache_size: usize,

    #[serde(default = "default_disk_kv_cache_size")]
    pub disk_kv_cache_size: usize,

    #[serde(default = "default_disk_kv_cache_path")]
    pub disk_kv_cache_path: String,

    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,

    #[serde(default = "default_tp_size")]
    pub tp_size: u8,
}
