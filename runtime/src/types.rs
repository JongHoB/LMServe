use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    All,
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum RoutePolicy {
    RoundRobin,
    LoadBalance,
}
