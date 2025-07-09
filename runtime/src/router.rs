use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use tonic::transport::{Channel, Endpoint, Uri};
use tracing::debug;

use crate::pb::llm::llm_client::LlmClient;
use crate::pb::llm::{
    GenerateRequest, GenerateResponse, KvAgentMetadata, ReserveRequest, ReserveResponse,
    TransferKvRequest, TriggerRequest,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EngineKind {
    All,
    Prefill,
    Decode,
}

impl FromStr for EngineKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "all" => Ok(EngineKind::All),
            "prefill" => Ok(EngineKind::Prefill),
            "decode" => Ok(EngineKind::Decode),
            _ => Err(format!("Unknown engine kind: {}", s)),
        }
    }
}

pub struct EngineRouter {
    mapping_table: Arc<Mutex<HashMap<EngineKind, VecDeque<Endpoint>>>>,
    kv_agent_table: Arc<RwLock<HashMap<String, KvAgentMetadata>>>,
}

impl Default for EngineRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineRouter {
    pub fn new() -> Self {
        Self {
            mapping_table: Arc::new(Mutex::new(HashMap::new())),
            kv_agent_table: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_node(&mut self, new_url: &str) -> Result<()> {
        let new_uri: String = new_url.parse().unwrap();
        let new_endpoint = Channel::from_shared(new_uri.clone().into_bytes())?;

        // TODO(jinu): Add timeout.
        let mut new_client = loop {
            match LlmClient::connect(new_endpoint.clone()).await {
                Ok(client) => {
                    break client;
                }
                Err(error) => {
                    debug!("Trying to connect to llm server ({}): {:?}", new_url, error);
                }
            };

            sleep(Duration::from_millis(500)).await;
        };

        let response = new_client.get_kind(()).await?.into_inner();
        let kind =
            EngineKind::from_str(&response.kind).unwrap_or_else(|e| panic!("Invalid kind: {e}"));

        let new_kv_agent_metadata = new_client.get_kv_agent_metadata(()).await?.into_inner();
        {
            let kv_agent_table_guard = self.kv_agent_table.read().await;

            // Notifies the newly added agent about all previously added agents.
            for kv_agent_metadata in kv_agent_table_guard.values() {
                new_client
                    .add_remote_kv_agent(kv_agent_metadata.clone())
                    .await?;
            }

            // Nofitifes all previously added agents about the newly added agent.
            for uri in kv_agent_table_guard.keys() {
                let endpoint = Channel::from_shared(uri.clone().into_bytes())?;
                let mut client = LlmClient::connect(endpoint).await?;

                client
                    .add_remote_kv_agent(new_kv_agent_metadata.clone())
                    .await?;
            }
        }

        {
            self.kv_agent_table
                .write()
                .await
                .insert(new_endpoint.uri().to_string(), new_kv_agent_metadata);
        }

        self.mapping_table
            .lock()
            .await
            .entry(kind)
            .and_modify(|endpoints| endpoints.push_back(new_endpoint.clone()))
            .or_insert(vec![new_endpoint].into());

        Ok(())
    }

    async fn get_client(&self, kind: EngineKind) -> Result<(LlmClient<Channel>, Uri)> {
        let endpoint = {
            let mut mapping_table_guard = self.mapping_table.lock().await;

            let endpoints = mapping_table_guard
                .get_mut(&kind)
                .ok_or_else(|| anyhow::anyhow!("No endpoints for {kind:?}"))?;

            endpoints.rotate_left(1);

            endpoints
                .front()
                .ok_or_else(|| anyhow::anyhow!("Endpoint list for {kind:?} is empty"))?
                .clone()
        };

        let uri = endpoint.uri().clone();

        Ok((LlmClient::connect(endpoint).await?, uri))
    }

    // FIXME(jinu)
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let response;
        if let Ok((mut client, _)) = self.get_client(EngineKind::All).await {
            response = client.generate(request).await?.into_inner();
        } else {
            response = self.generate_dist(request).await?;
        }

        Ok(response)
    }

    pub async fn generate_dist(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let (mut prefill_client, _) = self.get_client(EngineKind::Prefill).await?;
        let (mut decode_client, decode_uri) = self.get_client(EngineKind::Decode).await?;

        let num_samples = request.num_samples;
        let max_output_len = request.max_output_len;
        let ignore_eos = request.ignore_eos;
        let session_id = request.session_id;

        let p_gen_req = GenerateRequest {
            session_id: session_id.clone(),
            input_ids: request.input_ids.clone(),
            num_samples: 1,
            max_output_len: Some(1),
            ignore_eos,
        };

        let p_gen_res: GenerateResponse = prefill_client.generate(p_gen_req).await?.into_inner();

        let output_ids = p_gen_res.output_ids;
        let max_output_len = max_output_len.map(|max_output_len| max_output_len - 1);

        let new_input_ids = [request.input_ids.clone(), output_ids].concat();
        let d_reserve_req = ReserveRequest {
            session_id: session_id.clone(),
            input_ids: new_input_ids.clone(),
            num_samples,
            max_output_len,
            ignore_eos,
        };

        let d_reserve_res: ReserveResponse =
            decode_client.reserve(d_reserve_req).await?.into_inner();

        let success_hashes: Vec<u64> = if d_reserve_res.hash_values.is_empty() {
            Vec::new()
        } else {
            let kv_agent_table_guard = self.kv_agent_table.read().await;
            let kv_agent_meta_data = kv_agent_table_guard
                .get(&decode_uri.to_string())
                .unwrap_or_else(|| panic!("KV agent metadata not found for URI: {}", decode_uri));

            // FIXME(jinu)
            let peer_names = kv_agent_meta_data
                .agents
                .iter()
                .map(|agent| agent.agent_name.clone())
                .collect();

            let transfer_req = TransferKvRequest {
                session_id: session_id.clone(),
                hash_values: d_reserve_res.hash_values,
                peer_names,
                kv_descs: d_reserve_res.kv_descs,
            };

            let transfer_res = prefill_client.transfer_kv(transfer_req).await?.into_inner();
            transfer_res.success_hashes
        };

        let d_trg_req = TriggerRequest {
            session_id: session_id.clone(),
            hash_values: success_hashes,
        };

        let d_gen_res: GenerateResponse = decode_client.trigger(d_trg_req).await?.into_inner();
        Ok(d_gen_res)
    }

    pub async fn clear_cache(&self) -> Result<()> {
        {
            let mapping_table_guard = self.mapping_table.lock().await;

            for endpoints in mapping_table_guard.values() {
                for endpoint in endpoints {
                    let mut client = LlmClient::connect(endpoint.clone()).await?;
                    client.clear_cache(()).await?;
                }
            }
        }
        Ok(())
    }
}
