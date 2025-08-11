use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use tonic::transport::Channel;
use tracing::{debug, error, info, warn};

use crate::monitor::StatsMonitor;
use crate::pb::llm::llm_client::LlmClient;
use crate::pb::llm::{
    GenerateRequest, GenerateResponse, KvAgentMetadata, ReserveRequest, ReserveResponse,
    TransferKvRequest, TriggerRequest,
};
use crate::stats::Stats;
use crate::types::{EngineKind, RoutePolicy};

struct Node {
    id: String,
    kind: EngineKind,
    channel: Channel,
    kv_agent_metadata: KvAgentMetadata,
}

impl Node {
    async fn connect(&self) -> Result<LlmClient<Channel>> {
        Ok(LlmClient::new(self.channel.clone()))
    }
}

async fn establish_node(uri: &str) -> Result<Node> {
    let uri: String = uri.parse().unwrap();
    let endpoint = Channel::from_shared(uri.clone().into_bytes())?;

    let channel = loop {
        match endpoint.connect().await {
            Ok(client) => {
                break client;
            }
            Err(error) => {
                debug!("Trying to connect to llm server ({}): {:?}", uri, error);
            }
        };

        sleep(Duration::from_millis(500)).await;
    };

    let mut client = LlmClient::new(channel.clone());
    let info_response = client.get_info(()).await?.into_inner();

    let id = info_response.id;
    let kind =
        EngineKind::from_str(&info_response.kind).unwrap_or_else(|e| panic!("Invalid kind: {e}"));
    let kv_agent_metadata = client.get_kv_agent_metadata(()).await?.into_inner();

    Ok(Node {
        id,
        kind,
        channel,
        kv_agent_metadata,
    })
}

#[async_trait::async_trait]
trait NodeScheduler: Send + Sync {
    async fn add_node(&self, node: Node);

    async fn select(&self, request: &GenerateRequest) -> Result<Arc<Node>>;

    async fn get_nodes(&self) -> Vec<Arc<Node>>;

    async fn get_num_nodes(&self) -> usize;
}

struct RoundRobinNodeScheduler {
    nodes: Arc<Mutex<VecDeque<Arc<Node>>>>,
}

impl RoundRobinNodeScheduler {
    fn new() -> Self {
        Self {
            nodes: Default::default(),
        }
    }
}

#[async_trait::async_trait]
impl NodeScheduler for RoundRobinNodeScheduler {
    async fn add_node(&self, node: Node) {
        self.nodes.lock().await.push_back(Arc::new(node));
    }

    #[allow(unused_variables)]
    async fn select(&self, request: &GenerateRequest) -> Result<Arc<Node>> {
        let mut nodes_guard = self.nodes.lock().await;

        nodes_guard.rotate_right(1);

        let node = nodes_guard
            .front()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Routing failed: no available nodes"))?;

        Ok(node)
    }

    async fn get_nodes(&self) -> Vec<Arc<Node>> {
        let nodes_guard = self.nodes.lock().await;
        nodes_guard.iter().cloned().collect()
    }

    async fn get_num_nodes(&self) -> usize {
        self.nodes.lock().await.len()
    }
}

struct LoadBalanceNodeScheduler {
    node_map: Arc<RwLock<HashMap<String, Arc<Node>>>>,
    nats_client: async_nats::Client,
}

impl LoadBalanceNodeScheduler {
    fn new(nats_client: async_nats::Client) -> Self {
        Self {
            node_map: Default::default(),
            nats_client,
        }
    }
}

#[async_trait::async_trait]
impl NodeScheduler for LoadBalanceNodeScheduler {
    async fn add_node(&self, node: Node) {
        self.node_map
            .write()
            .await
            .insert(node.id.clone(), Arc::new(node));
    }

    #[allow(unused_variables)]
    async fn select(&self, request: &GenerateRequest) -> Result<Arc<Node>> {
        let node_map_guard = self.node_map.read().await;

        let ids: Vec<String> = node_map_guard.keys().cloned().collect();
        let payload = serde_json::to_vec(&ids)?;
        let msg = self
            .nats_client
            .request("stats.query.list", payload.into())
            .await?;

        let stats_map: HashMap<String, Stats> = serde_json::from_slice(&msg.payload)?;

        let id: &str = stats_map
            .iter()
            .min_by(|a, b| {
                let num_wait_reqs_a = a.1.num_waiting_reqs + a.1.num_pendding_reqs;
                let num_wait_reqs_b = b.1.num_waiting_reqs + b.1.num_pendding_reqs;

                let gpu_kv_usage_a = a.1.gpu_kv_block_usage;
                let gpu_kv_usage_b = b.1.gpu_kv_block_usage;

                let host_kv_usage_a = a.1.host_kv_block_usage;
                let host_kv_usage_b = b.1.host_kv_block_usage;

                num_wait_reqs_a.cmp(&num_wait_reqs_b).then(
                    gpu_kv_usage_a
                        .partial_cmp(&gpu_kv_usage_b)
                        .unwrap_or(Ordering::Equal),
                )
            })
            .map(|(id, _)| id)
            .ok_or_else(|| anyhow::anyhow!("Routing failed: no available nodes"))?;

        let node = node_map_guard
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Routing failed: no available nodes"))?;

        Ok(node)
    }

    async fn get_nodes(&self) -> Vec<Arc<Node>> {
        let node_map_guard = self.node_map.read().await;
        node_map_guard.values().cloned().collect()
    }

    async fn get_num_nodes(&self) -> usize {
        self.node_map.read().await.len()
    }
}

async fn get_node_scheduler(policy: RoutePolicy, nats_uri: String) -> Box<dyn NodeScheduler> {
    match policy {
        RoutePolicy::RoundRobin => Box::new(RoundRobinNodeScheduler::new()),
        RoutePolicy::LoadBalance => {
            let nc = match async_nats::connect(&nats_uri).await {
                Ok(nc) => Some(nc),
                Err(e) => {
                    warn!("Failed to connect nats server: {e}");
                    info!("NATS server connection failed; falling back to Round-Robin scheduler");
                    None
                }
            };

            match nc {
                Some(nc) => Box::new(LoadBalanceNodeScheduler::new(nc)),
                None => Box::new(RoundRobinNodeScheduler::new()),
            }
        }
    }
}

struct LlmRouter {
    scheduler: Box<dyn NodeScheduler>,
}

impl LlmRouter {
    async fn new(policy: RoutePolicy, nats_uri: String) -> Self {
        Self {
            scheduler: get_node_scheduler(policy, nats_uri).await,
        }
    }

    async fn add_node(&self, node: Node) {
        self.scheduler.add_node(node).await;
    }

    async fn get_num_nodes(&self) -> usize {
        self.scheduler.get_num_nodes().await
    }

    async fn handshake_kv_agent(&self, target_node: &Node) -> Result<()> {
        let mut target_client = target_node.connect().await?;

        // Register the new KV agent with existing agents, and vice versa.
        for node in self.scheduler.get_nodes().await {
            let mut client = node.connect().await?;
            client
                .add_remote_kv_agent(target_node.kv_agent_metadata.clone())
                .await?;

            target_client
                .add_remote_kv_agent(node.kv_agent_metadata.clone())
                .await?;
        }

        Ok(())
    }

    async fn route(&self, request: GenerateRequest) -> Result<(GenerateResponse, Arc<Node>)> {
        let node = self.scheduler.select(&request).await?;

        let mut client = node.connect().await?;
        let response = client.generate(request).await?.into_inner();

        Ok((response, node))
    }

    async fn forward(
        &self,
        src_node: &Node,
        request: GenerateRequest,
    ) -> Result<(GenerateResponse, Arc<Node>)> {
        let session_id = request.session_id.clone();
        let input_ids = request.input_ids.clone();
        let num_samples = request.num_samples;
        let max_output_len = request.max_output_len;
        let ignore_eos = request.ignore_eos;

        // TODO(jinu): Implement ReserveRequest::from_request()
        let reserve_req = ReserveRequest {
            session_id: session_id.clone(),
            input_ids,
            num_samples,
            max_output_len,
            ignore_eos,
        };

        let dst_node = self.scheduler.select(&request).await?;
        let mut dst_client = dst_node.connect().await?;

        let reserve_res: ReserveResponse = dst_client.reserve(reserve_req).await?.into_inner();

        let success_hashes: Vec<u64> = if reserve_res.hash_values.is_empty() {
            Vec::new()
        } else {
            // FIXME(jinu)
            let peer_names = dst_node
                .kv_agent_metadata
                .agents
                .iter()
                .map(|agent| agent.agent_name.clone())
                .collect();

            let transfer_req = TransferKvRequest {
                session_id: session_id.clone(),
                hash_values: reserve_res.hash_values,
                peer_names,
                kv_descs: reserve_res.kv_descs,
            };

            let mut src_client = src_node.connect().await?;
            let transfer_res = src_client.transfer_kv(transfer_req).await?.into_inner();
            transfer_res.success_hashes
        };
        let trg_req = TriggerRequest {
            session_id: session_id.clone(),
            hash_values: success_hashes,
        };

        let gen_res: GenerateResponse = dst_client.trigger(trg_req).await?.into_inner();

        Ok((gen_res, dst_node))
    }

    async fn clear_cache(&self) -> Result<()> {
        for node in self.scheduler.get_nodes().await {
            let mut client = node.connect().await?;
            client.clear_cache(()).await?;
        }

        Ok(())
    }
}

pub struct Controller {
    router: Arc<LlmRouter>,
    prefill_router: Arc<LlmRouter>,
    monitor: Option<StatsMonitor>,
}

impl Controller {
    pub async fn new(policy: RoutePolicy, nats_uri: String) -> Self {
        let monitor = match StatsMonitor::start(&nats_uri) {
            Ok(mon) => Some(mon),
            Err(e) => {
                warn!("Failed to connect nats server: {:?}", e);
                None
            }
        };

        Self {
            router: Arc::new(LlmRouter::new(policy, nats_uri.clone()).await),
            prefill_router: Arc::new(LlmRouter::new(policy, nats_uri.clone()).await),
            monitor,
        }
    }

    pub async fn add_node(&self, uri: &str) -> Result<()> {
        let node = establish_node(uri).await?;

        self.router.handshake_kv_agent(&node).await?;
        self.prefill_router.handshake_kv_agent(&node).await?;

        match node.kind {
            EngineKind::Prefill => {
                self.prefill_router.add_node(node).await;
            }
            _ => {
                self.router.add_node(node).await;
            }
        }

        Ok(())
    }

    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let response = if self.prefill_router.get_num_nodes().await > 0 {
            // P/D disaggregation serving
            let max_output_len = request.max_output_len;
            let mut p_gen_req = request.clone();
            let mut d_gen_req = request;

            p_gen_req.num_samples = 1;
            p_gen_req.max_output_len = Some(1);

            let (p_gen_res, p_node) = self.prefill_router.route(p_gen_req).await?;

            d_gen_req
                .input_ids
                .append(&mut p_gen_res.output_ids.clone());
            d_gen_req.max_output_len = max_output_len.map(|v| v.saturating_sub(1));

            let (mut d_gen_res, _) = self.router.forward(&p_node, d_gen_req).await?;

            let mut res = p_gen_res;
            res.output_ids.append(&mut d_gen_res.output_ids);
            res.token_latencies.append(&mut d_gen_res.token_latencies);

            res
        } else {
            let (res, _) = self.router.route(request).await?;

            res
        };

        Ok(response)
    }

    pub async fn clear_cache(&self) -> Result<()> {
        self.router.clear_cache().await?;
        self.prefill_router.clear_cache().await?;

        Ok(())
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        if let Some(mut monitor) = self.monitor.take() {
            monitor
                .shutdown()
                .unwrap_or_else(|e| error!("Failed to shutdown monitor daemon: {:?}", e));
        }
    }
}
