use std::process::{Child, Command};
use tracing::info;

use clap::Parser;
use serde::Serialize;
use serde_json::Value;

use api_server::args::APIServerArgs;
use launcher::args::LauncherArgs;
use llmserve::args::LLMEngineArgs;

fn spawn_process(path: &str, args: &[String]) -> std::io::Result<Child> {
    info!("Launching {}...", path);
    let child = Command::new(path).args(args).spawn()?;

    Ok(child)
}

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
    utils::logging::init_tracing();

    let args = LauncherArgs::parse();

    let model_name = args.model_name;

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

    let mut llmserve = spawn_process("llmserve", &to_cmd_args(&engine_args))?;

    let engine_url = format!("http://[::1]:{}", engine_args.port);

    let api_server_args = APIServerArgs {
        tokenier_name: model_name.clone(),
        engine_urls: vec![engine_url],
        address: args.address,
        port: args.port,
    };

    let mut api_server = spawn_process("api_server", &to_cmd_args(&api_server_args))?;

    utils::signal_handler::wait_shutdown_signal().await;

    api_server.wait()?;
    llmserve.wait()?;

    Ok(())
}
