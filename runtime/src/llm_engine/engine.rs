use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::join;
use tokio::sync::{Mutex, Notify};
use tracing::{info, warn};

use super::infer_task::InferTask;
use super::scheduler::Scheduler;
use super::sequence::{SeqStatus, Sequence};
use super::stub::LLMEngineStub;
use super::worker::{KVWorkerGroup, ModelWorkerGroup};

pub fn norm_log_probs(probs: &[f32]) -> f32 {
    let log_sum: f32 = probs
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
        .sum();

    log_sum / probs.len() as f32
}

type Bytes = Vec<u8>;

#[allow(dead_code)]
pub struct LLMEngine {
    model_name: String,
    block_size: usize,
    gpu_memory_fraction: f32,
    max_batch_size: usize,
    max_seq_len: usize,
    max_num_batched_tokens: usize,

    tp_size: u8,

    model_worker_group: ModelWorkerGroup,
    kv_worker_group: KVWorkerGroup,
    kv_agent_worker_group: KVWorkerGroup,

    kv_local_agent_metadata: Vec<Bytes>,
    kv_remote_agent_table: Arc<Mutex<HashMap<String, Vec<String>>>>,

    scheduler: Arc<Mutex<Scheduler>>,
}

impl LLMEngine {
    pub async fn new(
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        host_cache_size: usize,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
        worker_group_uds_path: String,
    ) -> Result<LLMEngine> {
        let model_worker_group = ModelWorkerGroup::init(tp_size, worker_group_uds_path.clone())
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize model worker group: {e}"));

        let (total, peak) = model_worker_group
            .warmup(max_batch_size, max_seq_len, max_num_batched_tokens)
            .await
            .unwrap_or_else(|e| panic!("Failed to warmup worker: {e}"));

        let available = (total as f32 * gpu_memory_fraction) as u64;
        let cache_size = available.saturating_sub(peak) as usize;
        info!(
            "Allocating {:.2} GB of memory per GPU for KV cache.",
            cache_size as f64 / (1 << 30) as f64
        );

        // Convert GBs -> Bytes
        let host_cache_size = host_cache_size * (1 << 30);

        let (num_gpu_blocks, num_host_blocks) = model_worker_group
            .init_cache(cache_size, host_cache_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to init KV cache: {e}"));
        info!("Created {num_gpu_blocks} KV cache blocks on GPU memory.");
        if num_host_blocks > 0 {
            info!("Created {num_host_blocks} KV cache blocks on Host memory.");
        }

        let kv_worker_group = KVWorkerGroup::init(tp_size, worker_group_uds_path.clone())
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize KV worker group: {e}"));

        let kv_agent_worker_group = KVWorkerGroup::init(tp_size, worker_group_uds_path.clone())
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize KV worker group: {e}"));

        let kv_local_agent_metadata = kv_agent_worker_group
            .get_local_agent_metadata()
            .await
            .unwrap_or_else(|e| panic!("Failed to get local agent metadata: {e}"));

        let scheduler = Scheduler::new(
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            block_size,
            num_gpu_blocks as usize,
            num_host_blocks as usize,
        );

        Ok(LLMEngine {
            model_name,
            block_size,
            gpu_memory_fraction,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tp_size,
            model_worker_group,
            kv_worker_group,
            kv_agent_worker_group,
            kv_local_agent_metadata,
            kv_remote_agent_table: Arc::new(Mutex::new(HashMap::new())),
            scheduler: Arc::new(Mutex::new(scheduler)),
        })
    }

    async fn add_request(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
        server_url: Option<String>,
    ) -> Result<()> {
        let mut seqs: Vec<Sequence> = Vec::new();
        for _ in 0..num_samples {
            let seq = Sequence::new(
                session_id.clone(),
                input_ids.clone(),
                max_output_len,
                ignore_eos,
            );
            seqs.push(seq);
        }

        let infer_task = InferTask::new(session_id.clone(), seqs, utils::time::now_ns());

        if let Some(server_url) = server_url {
            self.pull_task_data(&infer_task, server_url).await?;
        }

        {
            self.scheduler.lock().await.add(infer_task);
        }

        Ok(())
    }

    async fn iter(&self) -> Option<Vec<InferTask>> {
        let (infer_inputs, fetch_block_mappings, write_through_block_mappings) =
            { self.scheduler.lock().await.schedule() };
        if infer_inputs.is_empty() {
            return None;
        }

        let wait_before_execute = !fetch_block_mappings.is_empty();
        let record_after_execute = !write_through_block_mappings.is_empty();

        let (infer_result, transfer_result) = join!(
            self.model_worker_group
                .infer(infer_inputs, wait_before_execute, record_after_execute),
            self.kv_worker_group
                .transfer_kv(fetch_block_mappings, write_through_block_mappings),
        );

        let infer_outputs =
            infer_result.unwrap_or_else(|e| panic!("Failed to execute worker: {e}"));
        transfer_result.unwrap_or_else(|e| panic!("Failed to transfer KV: {e}"));

        let finished_tasks = { self.scheduler.lock().await.commit(infer_outputs) };

        if !finished_tasks.is_empty() {
            Some(finished_tasks)
        } else {
            None
        }
    }

    fn get_local_agent_metadata(&self) -> Vec<Bytes> {
        self.kv_local_agent_metadata.clone()
    }

    async fn check_remote_agent_metadata(&self, server_url: String) -> Option<Vec<String>> {
        let kv_agent_table = self.kv_remote_agent_table.lock().await;

        kv_agent_table.get(&server_url).cloned()
    }

    async fn add_remote_agent_metadata(
        &self,
        server_url: String,
        remote_agent_metadata: Vec<Bytes>,
    ) -> Result<Vec<String>> {
        let mut kv_agent_table = self.kv_remote_agent_table.lock().await;

        let peer_names = kv_agent_table
            .entry(server_url.clone())
            .or_insert_with(Vec::new);

        if !peer_names.is_empty() {
            return Ok(peer_names.clone());
        }

        let new_peer_names = self
            .kv_agent_worker_group
            .add_remote_agent_metadata(remote_agent_metadata)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to set KV agent metadata: {e}"))?;

        kv_agent_table.insert(server_url, new_peer_names.clone());

        Ok(new_peer_names)
    }

    async fn get_peer_names(&self, server_url: String) -> Result<Vec<String>> {
        let peer_names = match self.check_remote_agent_metadata(server_url.clone()).await {
            Some(peer_names) => peer_names,
            None => {
                let remote_kv_agent_metadata = self
                    .get_remote_kv_agent_metadata(server_url.clone())
                    .await?;

                self.add_remote_agent_metadata(server_url.clone(), remote_kv_agent_metadata)
                    .await?
            }
        };

        Ok(peer_names)
    }

    async fn pull_task_data(&self, infer_task: &InferTask, server_url: String) -> Result<()> {
        let peer_names = self.get_peer_names(server_url.clone()).await?;
        let seqs = infer_task.get_active_seqs();
        let head_seq = seqs.first().expect("No active sequence found");
        let token_ids = head_seq.get_token_ids().to_vec();

        let cached_token_len = {
            self.scheduler
                .lock()
                .await
                .get_host_cache_token_len(&token_ids)
        };

        let num_cached_blocks = cached_token_len / self.block_size;
        let max_pull_blocks = (token_ids.len() - 1) / self.block_size;

        if num_cached_blocks >= max_pull_blocks {
            return Ok(());
        }

        let (remote_descs, last_token_idx) = self
            .get_remote_descriptors(
                server_url.clone(),
                token_ids.clone(),
                cached_token_len,
                token_ids.len(),
            )
            .await?;

        if remote_descs.is_empty() {
            return Ok(());
        }

        let block_ids = {
            self.scheduler
                .lock()
                .await
                .hold_seq_tokens(head_seq, cached_token_len, last_token_idx)
        };

        if !self.pull_kv(peer_names, remote_descs, block_ids).await {
            warn!("Failed to pull remote task data; scheduling task without retrieved data");
        }

        {
            let mut scheduler = self.scheduler.lock().await;
            // Release sequence tokens that were held during KV pulling.
            scheduler.release_seq_tokens(head_seq);
            scheduler.init_prefix_host_cache_blocks(head_seq);
        }

        Ok(())
    }

    // TODO(jinu): Pin blocks corresponding to descriptors, and unpin them when transfer completion is notified.
    async fn get_descriptors(
        &self,
        token_ids: Vec<u32>,
        start: usize,
        end: usize,
    ) -> Result<(Vec<Bytes>, usize)> {
        let (block_ids, last_token_idx) = {
            self.scheduler
                .lock()
                .await
                .get_host_cache_block_range(token_ids, start, end)
        };

        if block_ids.is_empty() {
            return Ok((vec![], last_token_idx));
        }

        let descs = self
            .kv_agent_worker_group
            .get_descriptors(block_ids)
            .await
            .unwrap_or_else(|e| panic!("Failed to get descriptors: {e}"));

        Ok((descs, last_token_idx))
    }

    async fn get_remote_descriptors(
        &self,
        remote_url: String,
        token_ids: Vec<u32>,
        start: usize,
        end: usize,
    ) -> Result<(Vec<Bytes>, usize)> {
        let (descs, last_token_idx) =
            LLMEngineStub::get_descriptors(remote_url, token_ids, start, end).await?;

        Ok((descs, last_token_idx))
    }

    async fn get_remote_kv_agent_metadata(&self, remote_url: String) -> Result<Vec<Bytes>> {
        LLMEngineStub::get_remote_kv_agent_metadata(remote_url).await
    }

    async fn pull_kv(
        &self,
        peer_names: Vec<String>,
        remote_descs: Vec<Bytes>,
        block_ids: Vec<u32>,
    ) -> bool {
        self.kv_agent_worker_group
            .pull_kv(peer_names, remote_descs, block_ids)
            .await
            .unwrap_or_else(|e| panic!("Failed to pull KVs: {e}"))
    }
}

pub struct LLMEngineOutput {
    pub session_id: String,
    pub output_ids: Vec<u32>,
    /// Token latencies in seconds.
    pub token_latencies: Vec<f32>,
}

impl LLMEngineOutput {
    pub fn from_task(task: &InferTask) -> Self {
        let seqs = task.get_seqs(SeqStatus::Finished);
        let selected_seq = seqs
            .iter()
            .max_by(|a, b| {
                norm_log_probs(a.get_token_probs().as_ref())
                    .partial_cmp(&norm_log_probs(b.get_token_probs().as_ref()))
                    .unwrap()
            })
            .expect("Failed to select sequence: max_by() returned None.");

        let output_ids = selected_seq.get_output_ids();

        let mut token_times = Vec::with_capacity(selected_seq.append_token_times.len() + 1);
        token_times.push(task.get_arrival_time());
        token_times.extend_from_slice(&selected_seq.append_token_times);

        let token_latencies: Vec<f32> = token_times
            .windows(2)
            .map(|pair| pair[1].saturating_sub(pair[0]) as f32 / 1e9)
            .collect();

        Self {
            session_id: task.get_session_id(),
            output_ids: output_ids.to_vec(),
            token_latencies,
        }
    }
}

pub struct LLMEngineWrapper {
    engine: Arc<LLMEngine>,

    request_events: Mutex<HashMap<String, Arc<Notify>>>,
    request_outputs: Mutex<HashMap<String, InferTask>>,
}

impl LLMEngineWrapper {
    pub async fn new(
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        host_kv_cache_size: usize,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
        worker_group_uds_path: String,
    ) -> Result<LLMEngineWrapper> {
        let llm_engine = LLMEngine::new(
            model_name,
            block_size,
            gpu_memory_fraction,
            host_kv_cache_size,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tp_size,
            worker_group_uds_path,
        )
        .await?;

        Ok(LLMEngineWrapper {
            engine: Arc::new(llm_engine),
            request_events: Mutex::new(HashMap::new()),
            request_outputs: Mutex::new(HashMap::new()),
        })
    }

    pub async fn generate(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
        server_url: Option<String>,
    ) -> Result<LLMEngineOutput> {
        let notify = Arc::new(Notify::new());
        {
            self.request_events
                .lock()
                .await
                .insert(session_id.clone(), notify.clone());
        }

        self.engine
            .add_request(
                input_ids,
                num_samples,
                session_id.clone(),
                max_output_len,
                ignore_eos,
                server_url,
            )
            .await?;

        notify.notified().await;

        let request_output: InferTask = {
            self.request_outputs
                .lock()
                .await
                .remove(&session_id)
                .unwrap()
        };

        let engine_output = LLMEngineOutput::from_task(&request_output);
        Ok(engine_output)
    }

    pub async fn get_descriptors(
        &self,
        token_ids: Vec<u32>,
        start: usize,
        end: usize,
    ) -> Result<(Vec<Bytes>, usize)> {
        let (descs, last_token_idx) = self.engine.get_descriptors(token_ids, start, end).await?;

        Ok((descs, last_token_idx))
    }

    pub async fn get_kv_agent_metadata(&self) -> Result<Vec<Bytes>> {
        Ok(self.engine.get_local_agent_metadata())
    }

    pub async fn run_engine(&self) -> Result<()> {
        loop {
            let outputs = self.engine.iter().await;
            let Some(outputs) = outputs else {
                continue;
            };
            for infer_task in outputs.into_iter() {
                let session_id = infer_task.get_session_id();

                self.request_outputs
                    .lock()
                    .await
                    .insert(session_id.clone(), infer_task);

                let request_event = self
                    .request_events
                    .lock()
                    .await
                    .remove(&session_id)
                    .unwrap();
                request_event.notify_waiters()
            }
        }
    }
}
