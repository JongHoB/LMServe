use std::sync::Arc;

use anyhow::Result;
use tonic::{Request, Response, Status};

use crate::llm_engine::engine::LLMEngineWrapper;
use crate::types::EngineKind;

use crate::pb::llm::llm_server::{Llm, LlmServer};
use crate::pb::llm::{
    AgentMetadata, GenerateRequest, GenerateResponse, GetInfoResponse, GetStatusResponse,
    KvAgentMetadata, ReserveRequest, ReserveResponse, TransferKvRequest, TransferKvResponse,
    TriggerRequest,
};

pub struct LLMService {
    kind: EngineKind,
    engine: Arc<LLMEngineWrapper>,
}

impl LLMService {
    pub fn new(kind: EngineKind, engine: Arc<LLMEngineWrapper>) -> Self {
        Self { kind, engine }
    }

    pub fn into_server(self) -> LlmServer<Self> {
        LlmServer::new(self)
    }
}

#[tonic::async_trait]
impl Llm for LLMService {
    #[allow(unused_variables)]
    async fn get_info(&self, request: Request<()>) -> Result<Response<GetInfoResponse>, Status> {
        Ok(Response::new(GetInfoResponse {
            id: self.engine.get_id().await.to_string(),
            kind: self.kind.to_string(),
        }))
    }

    #[allow(unused_variables)]
    async fn get_stats(&self, request: Request<()>) -> Result<Response<GetStatusResponse>, Status> {
        let engine_status = self.engine.get_stats().await.expect("Failed to get status");

        Ok(Response::new(GetStatusResponse {
            engine_stats: Some(engine_status),
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
        let disable_cache = generate_request.disable_cache;

        let output = self
            .engine
            .generate(
                input_ids,
                num_samples as u16,
                session_id,
                max_output_len.map(|x| x as usize),
                ignore_eos,
                disable_cache,
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
        let disable_cache = reserve_request.disable_cache;

        let output = self
            .engine
            .reserve(
                input_ids,
                num_samples as u16,
                session_id,
                max_output_len.map(|x| x as usize),
                ignore_eos,
                disable_cache,
            )
            .await
            .expect("Failed to reserve request");

        return Ok(Response::new(ReserveResponse {
            region_id: output.region_id,
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
        let region_id = trigger_request.region_id;
        let hash_values = trigger_request.hash_values;

        let output = self
            .engine
            .trigger(session_id, region_id, hash_values)
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
