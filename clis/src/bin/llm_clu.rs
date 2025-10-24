use std::env;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::{Child, Command};

use clap::Parser;
use serde::Serialize;
use serde_json::Value;
use tracing::info;

use clis::args::{APIServerArgs, CLIArgs, LLMSrvArgs};
use clis::configs::{APIServerConfig, LLMCluConfig, LLMSrvConfig};
use runtime::configs::{ControllerConfig, EngineConfig};
use runtime::types;

fn to_cmd_args<T: Serialize>(args: &T) -> Vec<String> {
    let value = serde_json::to_value(args).expect("serialization failed");
    let obj = value.as_object().expect("expected a struct-like object");

    obj.iter()
        .filter_map(|(k, v)| {
            let key = k.replace("_", "-");
            match v {
                Value::Bool(false) => None,
                Value::Bool(true) => Some(format!("--{}", key)),
                Value::Number(n) => Some(format!("--{}={}", key, n)),
                Value::String(s) => Some(format!("--{}={}", key, s)),
                Value::Array(vec) => {
                    let vec: Vec<String> = vec
                        .iter()
                        .map(|v| match v {
                            Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        })
                        .collect();
                    Some(format!("--{}={}", key, vec.join(" ")))
                }
                _ => Some(format!("--{}={}", key, v)),
            }
        })
        .collect()
}

const LLMSERVE_BASE_PORT: u32 = 7000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_dir = env::var("LLMSERVE_HOME").expect("LLMSERVE_HOME is not set");
    let bin_path: PathBuf = PathBuf::from(root_dir).join("bin");

    if !bin_path.exists() {
        eprintln!(
            "Error: bin directory does not exist at {}",
            bin_path.display()
        );
    }

    utils::logging::init_tracing();

    let args = CLIArgs::parse();

    let config: LLMCluConfig = match args.config {
        Some(config) => {
            let file = fs::File::open(config)?;
            let reader = BufReader::new(file);

            serde_yaml::from_reader(reader)?
        }
        None => {
            let api_server_config = APIServerConfig {
                api_address: args.api_address.clone(),
            };

            let controller_config: ControllerConfig =
                serde_yaml::from_value(serde_yaml::to_value(&args)?)?;

            let mut llm_server_configs = Vec::with_capacity(args.num_worker_groups as usize);
            for i in 0..args.num_worker_groups {
                let port = LLMSERVE_BASE_PORT + i;

                let engine_config: EngineConfig =
                    serde_yaml::from_value(serde_yaml::to_value(&args)?)?;

                llm_server_configs.push(LLMSrvConfig {
                    kind: types::EngineKind::All,
                    address: format!("127.0.0.1:{port}"),
                    engine: engine_config,
                });
            }

            LLMCluConfig {
                model_name: args.model_name,
                api_server: api_server_config,
                controller: controller_config,
                llm_servers: llm_server_configs,
                nats_uri: args.nats_uri.clone(),
            }
        }
    };

    let model_name = config.model_name;
    let nats_uri = args.nats_uri.clone();

    let mut device_offset: u8 = 0;
    let mut llm_servers: Vec<Child> = Vec::with_capacity(config.llm_servers.len() as usize);
    for (group_id, llm_server) in config.llm_servers.iter().enumerate() {
        let devices: Vec<u8> =
            (device_offset..(device_offset + llm_server.engine.tp_size)).collect();

        let llm_srv_args = LLMSrvArgs {
            kind: llm_server.kind,
            model_name: model_name.clone(),
            engine: llm_server.engine.clone(),
            address: llm_server.address.clone(),
            devices: Some(devices),
            nats_uri: nats_uri.clone(),
        };

        info!("Launching llm_srv (group_id = {group_id})...");
        let engine = Command::new(bin_path.join("llm_srv"))
            .args(to_cmd_args(&llm_srv_args))
            .env("GROUP_ID", group_id.to_string())
            .spawn()?;

        llm_servers.push(engine);
        device_offset += llm_server.engine.tp_size;
    }

    let llm_server_addresses = config
        .llm_servers
        .iter()
        .map(|x| x.address.clone())
        .collect();

    let api_server_args = APIServerArgs {
        model_name: model_name.clone(),
        controller: config.controller,
        llm_server_addresses,
        api_address: config.api_server.api_address,
        nats_uri: nats_uri.clone(),
    };

    info!("Launching api_server...");
    let mut api_server = Command::new(bin_path.join("api_server"))
        .args(to_cmd_args(&api_server_args))
        .spawn()?;

    utils::signal_handler::wait_shutdown_signal().await;

    for mut llm_server in llm_servers {
        llm_server.wait()?;
    }
    api_server.wait()?;

    Ok(())
}
