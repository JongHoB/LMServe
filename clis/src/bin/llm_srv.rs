use std::collections::HashMap;
use std::env;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::path::Path;
use std::process::{Child, Command};
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use rand::seq::SliceRandom;
use tonic::transport::Server;
use tracing::{debug, info};
use tracing_futures::Instrument;

use runtime::llm_engine::engine::LLMEngineWrapper;
use runtime::llm_srv::LLMService;

use clis::args::LLMSrvArgs;

const WORKER_GROUP_UDS_PATH_PREFIX: &str = "/tmp/llmserve/group";

fn random_available_port(range: std::ops::Range<u16>) -> Option<u16> {
    let mut ports: Vec<u16> = range.collect();
    ports.shuffle(&mut rand::rng());

    for port in ports {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port);
        if TcpListener::bind(addr).is_ok() {
            return Some(port);
        }
    }
    None
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::logging::init_tracing();

    let args = LLMSrvArgs::parse();

    let address: SocketAddr = args.address.parse().unwrap_or_else(|_| {
        panic!(
            "Invalid address '{}'. Expected format: <host>:<port>",
            &args.address
        )
    });

    let root_span = tracing::info_span!("LLMServer", port=%address.port(), kind= %args.kind);
    let _root_guard = root_span.clone().entered();

    let group_id = env::var("GROUP_ID").unwrap_or(String::from("0"));
    let devices = args.devices.unwrap_or((0..args.tp_size).collect());

    let worker_group_uds_path = format!("{}-{}", WORKER_GROUP_UDS_PATH_PREFIX, group_id);
    let worker_port = random_available_port(6000..7000).expect("Not found available port");

    let mut workers: Vec<Child> = Vec::new();
    for (device, rank) in devices.iter().zip(0..args.tp_size) {
        let uds_path = format!("{}/model-{}", worker_group_uds_path, rank);

        std::fs::create_dir_all(Path::new(&uds_path).parent().unwrap())?;
        if Path::new(&uds_path).exists() {
            fs::remove_file(Path::new(&uds_path))
                .expect("Failed to remove existing UDS socket file");
        }

        let worker_args = vec![
            args.model_name.clone(),
            args.block_size.to_string(),
            device.to_string(),
            uds_path.to_string(),
        ];

        let mut worker_envs = HashMap::new();
        worker_envs.insert("RANK", rank.to_string());
        worker_envs.insert("WORLD_SIZE", args.tp_size.to_string());
        worker_envs.insert("MASTER_ADDR", address.ip().to_string());
        worker_envs.insert("MASTER_PORT", worker_port.to_string());

        info!("Launching llm-worker (group_id = {group_id}, rank = {rank})...");
        let worker = Command::new("llm-worker")
            .args(worker_args)
            .envs(worker_envs)
            .spawn()?;

        workers.push(worker);
    }

    let engine = Arc::new(
        LLMEngineWrapper::new(
            group_id,
            args.model_name,
            args.block_size,
            args.gpu_memory_fraction,
            args.host_kv_cache_size,
            args.max_batch_size,
            args.max_seq_len,
            args.max_num_batched_tokens,
            args.tp_size,
            worker_group_uds_path,
            args.nats_uri,
        )
        .await
        .expect("Failed to start API Server"),
    );

    // Run the engine asynchronously in the background
    let engine_clone = engine.clone();
    tokio::spawn(
        async move {
            engine_clone.run_engine().await.unwrap();
        }
        .instrument(root_span.clone()),
    );

    let llm_service = LLMService::new(args.kind, engine.clone());
    let svc = llm_service.into_server();

    debug!("LLMServer listening on: {address}");

    Server::builder()
        .add_service(svc)
        .serve_with_shutdown(address, utils::signal_handler::wait_shutdown_signal())
        .instrument(root_span.clone())
        .await?;

    for mut worker in workers {
        worker.wait()?;
    }

    Ok(())
}
