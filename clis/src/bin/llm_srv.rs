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
use tonic::{Request, Response, Status};
use tracing::{debug, info};
use tracing_futures::Instrument;

use runtime::llm_engine::engine::LLMEngineWrapper;
use runtime::types::EngineKind;

use clis::pb::llm::llm_server::{Llm, LlmServer};
use clis::pb::llm::{
    AgentMetadata, EngineStatus, GenerateRequest, GenerateResponse, GetKindResponse,
    GetStatusResponse, KvAgentMetadata, ReserveRequest, ReserveResponse, TransferKvRequest,
    TransferKvResponse, TriggerRequest,
};

use clis::args::LLMSrvArgs;

const WORKER_GROUP_UDS_PATH_PREFIX: &str = "/tmp/llmserve/group";

pub struct LLMService {
    kind: EngineKind,
    engine: Arc<LLMEngineWrapper>,
}

#[tonic::async_trait]
impl Llm for LLMService {
    #[allow(unused_variables)]
    async fn get_kind(&self, request: Request<()>) -> Result<Response<GetKindResponse>, Status> {
        Ok(Response::new(GetKindResponse {
            kind: self.kind.to_string(),
        }))
    }

    #[allow(unused_variables)]
    async fn get_status(
        &self,
        request: Request<()>,
    ) -> Result<Response<GetStatusResponse>, Status> {
        let engine_status = self
            .engine
            .get_status()
            .await
            .expect("Failed to get status");

        let status = EngineStatus {
            num_running_reqs: engine_status.num_running_reqs as u64,
            num_allocated_reqs: engine_status.num_allocated_reqs as u64,
            num_waiting_reqs: engine_status.num_waiting_reqs as u64,
            num_pendding_reqs: engine_status.num_pendding_reqs as u64,
            gpu_kv_block_usage: engine_status.gpu_kv_block_usage,
            host_kv_block_usage: engine_status.host_kv_block_usage,
        };

        Ok(Response::new(GetStatusResponse {
            engine_status: Some(status),
        }))
    }

    async fn add_remote_kv_agent(
        &self,
        request: Request<KvAgentMetadata>,
    ) -> Result<Response<()>, Status> {
        let agents = request.into_inner().agents;
        let kv_agent_metadata = agents
            .into_iter()
            .map(|agent| (agent.agent_name, agent.data))
            .collect();

        self.engine
            .add_remote_agent(kv_agent_metadata)
            .await
            .expect("Failed to add remote kv agent");

        Ok(Response::new(()))
    }

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let generate_request = request.into_inner();

        let session_id = generate_request.session_id;
        let input_ids = generate_request.input_ids;
        let num_samples = generate_request.num_samples;
        let max_output_len = generate_request.max_output_len;
        let ignore_eos = generate_request.ignore_eos;

        let output = self
            .engine
            .generate(
                input_ids,
                num_samples as u16,
                session_id,
                max_output_len.map(|x| x as usize),
                ignore_eos,
            )
            .await
            .expect("Failed to generate");

        Ok(Response::new(GenerateResponse {
            session_id: output.session_id,
            output_ids: output.output_ids,
            token_latencies: output.token_latencies,
        }))
    }

    async fn reserve(
        &self,
        request: Request<ReserveRequest>,
    ) -> Result<Response<ReserveResponse>, Status> {
        let reserve_request = request.into_inner();

        let session_id = reserve_request.session_id;
        let input_ids = reserve_request.input_ids;
        let num_samples = reserve_request.num_samples;
        let max_output_len = reserve_request.max_output_len;
        let ignore_eos = reserve_request.ignore_eos;

        let output = self
            .engine
            .reserve(
                input_ids,
                num_samples as u16,
                session_id,
                max_output_len.map(|x| x as usize),
                ignore_eos,
            )
            .await
            .expect("Failed to reserve request");

        return Ok(Response::new(ReserveResponse {
            hash_values: output.hash_values,
            kv_descs: output.kv_descs,
        }));
    }

    async fn transfer_kv(
        &self,
        request: Request<TransferKvRequest>,
    ) -> Result<Response<TransferKvResponse>, Status> {
        let transfer_request = request.into_inner();

        let session_id = transfer_request.session_id;
        let peer_names = transfer_request.peer_names;
        let kv_descs = transfer_request.kv_descs;
        let hash_values = transfer_request.hash_values;

        let output = self
            .engine
            .transfer_kv(session_id, peer_names, kv_descs, hash_values)
            .await
            .expect("Failed to transfer KV");

        Ok(Response::new(TransferKvResponse {
            success_hashes: output.hash_values,
        }))
    }

    async fn trigger(
        &self,
        request: Request<TriggerRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let trigger_request = request.into_inner();

        let session_id = trigger_request.session_id;
        let hash_values = trigger_request.hash_values;

        let output = self
            .engine
            .trigger(session_id, hash_values)
            .await
            .expect("Failed to trigger");

        Ok(Response::new(GenerateResponse {
            session_id: output.session_id,
            output_ids: output.output_ids,
            token_latencies: output.token_latencies,
        }))
    }

    #[allow(unused_variables)]
    async fn get_kv_agent_metadata(
        &self,
        request: Request<()>,
    ) -> Result<Response<KvAgentMetadata>, Status> {
        let local_agent_metadata = self
            .engine
            .get_kv_agent_metadata()
            .await
            .expect("Failed to get KV agent metadata");

        // FIXME(jinu)
        let agents = local_agent_metadata
            .into_iter()
            .map(|(agent_name, data)| AgentMetadata { agent_name, data })
            .collect();

        Ok(Response::new(KvAgentMetadata { agents }))
    }

    #[allow(unused_variables)]
    async fn clear_cache(&self, request: Request<()>) -> Result<Response<()>, Status> {
        self.engine
            .clear_cache()
            .await
            .expect("Failed to clear cache");

        Ok(Response::new(()))
    }
}

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
            args.model_name,
            args.block_size,
            args.gpu_memory_fraction,
            args.host_kv_cache_size,
            args.max_batch_size,
            args.max_seq_len,
            args.max_num_batched_tokens,
            args.tp_size,
            worker_group_uds_path,
        )
        .await
        .expect("Failed to start API Server"),
    );

    let llm_service = LLMService {
        kind: args.kind,
        engine: engine.clone(),
    };

    // Run the engine asynchronously in the background
    let engine_clone = engine.clone();
    tokio::spawn(
        async move {
            engine_clone.run_engine().await.unwrap();
        }
        .instrument(root_span.clone()),
    );

    let svc = LlmServer::new(llm_service);

    debug!("LLMServer listening on: {address}");

    Server::builder()
        .add_service(svc)
        .serve_with_shutdown(address, utils::signal_handler::wait_shutdown_signal())
        .await?;

    for mut worker in workers {
        worker.wait()?;
    }

    Ok(())
}
