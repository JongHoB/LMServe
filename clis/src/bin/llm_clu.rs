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

fn to_cmd_args<T: Serialize>(args: &T) -> Vec<String> {
    let value = serde_json::to_value(args).expect("serialization failed");
    let obj = value.as_object().expect("expected a struct-like object");

    obj.iter()
        .map(|(k, v)| {
            let key = k.replace("_", "-");
            let val = match v {
                Value::String(s) => s.clone(),
                Value::Array(vec) => {
                    let vec: Vec<String> = vec
                        .iter()
                        .map(|v| match v {
                            Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        })
                        .collect();
                    vec.join(" ")
                }
                _ => v.to_string(),
            };
            format!("--{}={}", key, val)
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
                address: args.address.clone(),
            };

            let llm_server_configs = (0..args.num_worker_groups)
                .map(|i| {
                    let port = LLMSERVE_BASE_PORT + i;

                    LLMSrvConfig {
                        kind: "full".to_string(),
                        block_size: args.block_size,
                        gpu_memory_fraction: args.gpu_memory_fraction,
                        host_kv_cache_size: args.host_kv_cache_size,
                        max_batch_size: args.max_batch_size,
                        max_seq_len: args.max_seq_len,
                        max_num_batched_tokens: args.max_num_batched_tokens,
                        tp_size: args.tp_size,
                        address: format!("127.0.0.1:{port}"),
                    }
                })
                .collect();

            LLMCluConfig {
                model_name: args.model_name,
                api_server: api_server_config,
                llm_servers: llm_server_configs,
            }
        }
    };

    let model_name = config.model_name;

    let mut device_offset: u8 = 0;
    let mut llm_servers: Vec<Child> = Vec::with_capacity(config.llm_servers.len() as usize);
    for (group_id, llm_server) in config.llm_servers.iter().enumerate() {
        let devices: Vec<u8> = (device_offset..(device_offset + llm_server.tp_size)).collect();
        let llm_srv_args = LLMSrvArgs {
            kind: llm_server.kind.clone(),
            model_name: model_name.clone(),
            block_size: llm_server.block_size,
            gpu_memory_fraction: llm_server.gpu_memory_fraction,
            host_kv_cache_size: llm_server.host_kv_cache_size,
            max_batch_size: llm_server.max_batch_size,
            max_seq_len: llm_server.max_seq_len,
            max_num_batched_tokens: llm_server.max_num_batched_tokens,
            tp_size: llm_server.tp_size,
            address: llm_server.address.clone(),
            devices: Some(devices),
        };

        info!("Launching llm_srv (group_id = {group_id})...");
        let engine = Command::new(bin_path.join("llm_srv"))
            .args(to_cmd_args(&llm_srv_args))
            .env("GROUP_ID", group_id.to_string())
            .spawn()?;

        llm_servers.push(engine);
        device_offset += llm_server.tp_size;
    }

    let llm_server_addresses = config
        .llm_servers
        .iter()
        .map(|x| x.address.clone())
        .collect();

    let api_server_args = APIServerArgs {
        model_name: model_name.clone(),
        address: config.api_server.address,
        llm_server_addresses,
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
