use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::{Child, Command};

use clap::Parser;
use serde::Serialize;
use serde_json::Value;

use api_server::args::APIServerArgs;
use launcher::args::LauncherArgs;
use llmserve::args::LLMEngineArgs;

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

const WORKER_UDS_PATH_PREFIX: &str = "/tmp/llmserve/worker";

const MASTER_ADDR: &str = "localhost";
const MASTER_PORT: u32 = 6000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::logging::init_tracing();

    let args = LauncherArgs::parse();

    let model_name = args.model_name;

    let mut workers: Vec<Child> = Vec::new();
    std::fs::create_dir_all(Path::new(WORKER_UDS_PATH_PREFIX).parent().unwrap())?;

    for rank in 0..args.tp_size {
        let uds_path = format!("{}-{}", WORKER_UDS_PATH_PREFIX, rank);
        if Path::new(&uds_path).exists() {
            fs::remove_file(Path::new(&uds_path))
                .expect("Failed to remove existing UDS socket file");
        }

        let worker_args = vec![
            model_name.clone(),
            args.block_size.to_string(),
            /*device=*/ workers.len().to_string(),
            uds_path.to_string(),
        ];

        let mut worker_envs = HashMap::new();
        worker_envs.insert("RANK", rank.to_string());
        worker_envs.insert("WORLD_SIZE", args.tp_size.to_string());
        worker_envs.insert("MASTER_ADDR", MASTER_ADDR.to_string());
        worker_envs.insert("MASTER_PORT", MASTER_PORT.to_string());

        let worker = Command::new("llmserve-worker")
            .args(worker_args)
            .envs(worker_envs)
            .spawn()?;
        workers.push(worker);
    }

    let engine_args = LLMEngineArgs {
        model_name: model_name.clone(),
        block_size: args.block_size,
        gpu_memory_fraction: args.gpu_memory_fraction,
        max_batch_size: args.max_batch_size,
        max_seq_len: args.max_seq_len,
        max_num_batched_tokens: args.max_num_batched_tokens,
        tp_size: args.tp_size,
        port: LLMSERVE_BASE_PORT,
    };

    let mut engine_envs = HashMap::new();
    engine_envs.insert("WORKER_UDS_PATH_PREFIX", WORKER_UDS_PATH_PREFIX);

    let mut llmserve = Command::new("llmserve")
        .args(&to_cmd_args(&engine_args))
        .envs(engine_envs)
        .spawn()?;

    let engine_url = format!("http://[::1]:{}", engine_args.port);

    let api_server_args = APIServerArgs {
        tokenier_name: model_name.clone(),
        engine_urls: vec![engine_url],
        address: args.address,
        port: args.port,
    };

    let mut api_server = Command::new("api_server")
        .args(&to_cmd_args(&api_server_args))
        .spawn()?;

    utils::signal_handler::wait_shutdown_signal().await;

    for mut worker in workers {
        worker.wait()?;
    }
    api_server.wait()?;
    llmserve.wait()?;

    Ok(())
}
