use runtime::configs::{ControllerConfig, EngineConfig};
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
    default_model_name,
    String,
    String::from("Qwen/Qwen2.5-0.5B")
);
default_fn!(default_api_address, String, String::from("127.0.0.1:8000"));
default_fn!(default_srv_address, String, String::from("127.0.0.1:7000"));
default_fn!(
    default_nats_uri,
    String,
    String::from("nats://127.0.0.1:4222")
);
default_fn!(
    default_engine_config,
    EngineConfig,
    serde_yaml::from_str::<EngineConfig>("{}").unwrap_or_else(|e| panic!("{e:?}"))
);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LLMCluConfig {
    pub model_name: String,

    #[serde(default = "default_nats_uri")]
    pub nats_uri: String,

    pub api_server: APIServerConfig,

    pub controller: ControllerConfig,

    pub llm_servers: Vec<LLMSrvConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct APIServerConfig {
    #[serde(default = "default_api_address")]
    pub api_address: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LLMSrvConfig {
    pub kind: types::EngineKind,

    pub address: String,

    #[serde(default = "default_engine_config")]
    pub engine: EngineConfig,
}
