use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::join;
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::stats::Stats;

use super::Bytes;
use super::infer_task::InferTask;
use super::outputs::{EngineStats, GenerateOutput, ReserveOutput, TransferOutput};
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use super::worker::{KVWorkerGroup, ModelWorkerGroup};

type AsyncResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
type AsyncTask = JoinHandle<AsyncResult>;
type AsyncQueue = Arc<Mutex<VecDeque<AsyncTask>>>;

struct Observer {
    handles: AsyncQueue,
    cancel: CancellationToken,
}

impl Default for Observer {
    fn default() -> Self {
        Self {
            handles: AsyncQueue::default(),
            cancel: CancellationToken::new(),
        }
    }
}

impl Observer {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    async fn register(&self, handle: AsyncTask) {
        self.handles.lock().await.push_back(handle);
    }

    async fn check_once(&self) -> usize {
        let mut finished: Vec<AsyncTask> = Vec::new();
        {
            let mut q = self.handles.lock().await;

            let mut i = 0;
            while i < q.len() {
                if q[i].is_finished() {
                    finished.push(q.remove(i).unwrap());
                } else {
                    i += 1;
                }
            }
        }

        let mut drained = 0;
        for h in finished {
            match h.await {
                Ok(Ok(())) => drained += 1,
                Ok(Err(e)) => {
                    eprintln!("[job error] {e}");
                    drained += 1;
                }
                Err(join_err) => {
                    eprintln!("[join error] {join_err}");
                    drained += 1;
                }
            }
        }

        drained
    }

    fn spawn_loop(self: &Arc<Self>, period: Duration) -> JoinHandle<()> {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = time::interval(period);
            loop {
                tokio::select! {
                    _ = this.cancel.cancelled() => break,
                    _ = tick.tick() => {
                        let n = this.check_once().await;
                        if n>0 {
                            //
                        }
                    }
                }
            }
        })
    }

    pub async fn cancel_all(&self) -> usize{
        let mut q = self.handles.lock().await;
        let n = q.len();
        for h in q.drain(..) {
            h.abort();
            let _ = h.await;
        }
        n
    }

    pub fn shutdown(&self) {
        self.cancel.cancel();
    }
}

#[allow(dead_code)]
pub struct LLMEngine {
    id: String,
    model_name: String,
    block_size: usize,
    gpu_memory_fraction: f32,
    max_batch_size: usize,
    max_seq_len: usize,
    max_num_batched_tokens: usize,

    tp_size: u8,

    model_worker_group: ModelWorkerGroup,
    kv_worker_group: KVWorkerGroup,
    kv_disk_worker_group: Arc<KVWorkerGroup>,
    kv_agent_worker_group: KVWorkerGroup,

    nats_client: Option<async_nats::Client>,

    local_kv_agent_metadata: Vec<(String, Bytes)>,

    scheduler: Arc<Mutex<Scheduler>>,

    observer: Arc<Observer>,
}

impl LLMEngine {
    pub async fn new(
        id: String,
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        host_cache_size: usize,
        disk_cache_size: usize,
        disk_cache_path: String,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
        worker_group_uds_path: String,
        nats_uri: String,
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
        let disk_cache_size = disk_cache_size * (1 << 30);

        let (num_gpu_blocks, num_host_blocks, num_disk_blocks) = model_worker_group
            .init_cache(
                cache_size,
                host_cache_size,
                disk_cache_size,
                disk_cache_path.clone(),
            )
            .await
            .unwrap_or_else(|e| panic!("Failed to init KV cache: {e}"));
        info!("Created {num_gpu_blocks} KV cache blocks on GPU memory.");
        if num_host_blocks > 0 {
            info!("Created {num_host_blocks} KV cache blocks on Host memory.");
        }
        if num_disk_blocks > 0 {
            info!(
                "Created {num_disk_blocks} KV cache blocks on Disk (path={}).",
                disk_cache_path,
            );
        }

        let kv_worker_group = KVWorkerGroup::init(tp_size, worker_group_uds_path.clone())
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize KV worker group: {e}"));

        let kv_disk_worker_group = KVWorkerGroup::init(tp_size, worker_group_uds_path.clone())
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize KV worker group: {e}"));

        let kv_agent_worker_group = KVWorkerGroup::init(tp_size, worker_group_uds_path.clone())
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize KV worker group: {e}"));

        let nats_client = match async_nats::connect(&nats_uri).await {
            Ok(nc) => Some(nc),
            Err(e) => {
                warn!("Failed to connect nats server: {e}");
                None
            }
        };

        let local_kv_agent_metadata = kv_agent_worker_group
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
            num_disk_blocks as usize,
        );

        // Publish stats to the NATS server for initial engine registration.
        let stats = scheduler.get_stats();
        if let Some(nc) = &nats_client {
            let subject: String = format!("stats.update.{}", id);
            nc.publish(subject, serde_json::to_vec(&stats)?.into())
                .await?
        }

        let observer = Observer::new();
        observer.spawn_loop(Duration::from_millis(200));

        Ok(LLMEngine {
            id,
            model_name,
            block_size,
            gpu_memory_fraction,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tp_size,
            model_worker_group,
            kv_worker_group,
            kv_disk_worker_group: Arc::new(kv_disk_worker_group),
            kv_agent_worker_group,
            nats_client,
            local_kv_agent_metadata,
            scheduler: Arc::new(Mutex::new(scheduler)),
            observer,
        })
    }

    async fn get_stats(&self) -> Stats {
        self.scheduler.lock().await.get_stats()
    }

    async fn publish_stats(&self, stats: Stats) -> Result<()> {
        match &self.nats_client {
            Some(nc) => {
                let subject: String = format!("stats.update.{}", self.id);
                nc.publish(subject, serde_json::to_vec(&stats)?.into())
                    .await
                    .map_err(Into::into)
            }
            None => Ok(()),
        }
    }

    async fn add_request(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
    ) -> Result<()> {
        let arrival_time = utils::time::now_ns();
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

        let infer_task = InferTask::new(session_id.clone(), seqs, arrival_time);
        let (plan, stats) = {
            let mut scheduler_guard = self.scheduler.lock().await;
            scheduler_guard.init_prefix_cache_blocks(&infer_task);

            let plan = scheduler_guard.plan_stage(&infer_task);

            match plan {
                Some(_) => {
                    scheduler_guard.pend(infer_task);
                }
                None => {
                    scheduler_guard.add(infer_task);
                }
            }

            let stats = scheduler_guard.get_stats();
            (plan, stats)
        };

        self.publish_stats(stats).await?;

        if let Some((block_mapping, hash_values)) = plan {
            let kv_disk_worker_group = Arc::clone(&self.kv_disk_worker_group);
            let scheduler = Arc::clone(&self.scheduler);
            let handle = tokio::spawn(async move {
                let _ = kv_disk_worker_group.swap_in_kv(vec![block_mapping]).await;
                scheduler
                    .lock()
                    .await
                    .trigger_pend_task(session_id, &hash_values);
                Ok(())
            });
            self.observer.register(handle).await;
        }

        Ok(())
    }

    async fn reserve_request(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
    ) -> Result<(Vec<Bytes>, Vec<u64>)> {
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

        let (block_ids, hash_values) = self.scheduler.lock().await.reserve_buffer(&infer_task);

        let kv_descs: Vec<Bytes> = if block_ids.is_empty() {
            Vec::new()
        } else {
            self.kv_agent_worker_group
                .get_descriptors(block_ids.to_vec())
                .await
                .unwrap_or_else(|e| panic!("Failed to get descriptors: {e}"))
        };

        self.scheduler.lock().await.pend(infer_task);

        Ok((kv_descs, hash_values))
    }

    async fn trigger_request(&self, session_id: String, hash_values: Vec<u64>) -> Result<()> {
        self.scheduler
            .lock()
            .await
            .trigger_pend_task(session_id, &hash_values);

        Ok(())
    }

    async fn iter(&self) -> Option<Vec<InferTask>> {
        let (infer_inputs, fetch_block_mappings, write_through_block_mappings, stats) = {
            // Lock scheduler briefly to check and schedule tasks.
            let mut scheduler_guard = self.scheduler.lock().await;
            if scheduler_guard.is_task_queue_empty() {
                return None;
            }

            scheduler_guard.schedule()
        };

        if infer_inputs.is_empty() {
            return None;
        }

        let wait_before_execute = !fetch_block_mappings.is_empty();
        let record_after_execute = !write_through_block_mappings.is_empty();

        let (infer_result, transfer_result, _) = join!(
            self.model_worker_group
                .infer(infer_inputs, wait_before_execute, record_after_execute),
            self.kv_worker_group
                .transfer_kv(fetch_block_mappings, write_through_block_mappings),
            self.publish_stats(stats),
        );

        let infer_outputs =
            infer_result.unwrap_or_else(|e| panic!("Failed to execute worker: {e}"));
        transfer_result.unwrap_or_else(|e| panic!("Failed to transfer KV: {e}"));

        let (finished_tasks, stats) = {
            let mut scheduler_guard = self.scheduler.lock().await;
            let finished_tasks = scheduler_guard.commit(infer_outputs);
            let stats = scheduler_guard.get_stats();
            (finished_tasks, stats)
        };

        if !finished_tasks.is_empty() {
            let _ = self.publish_stats(stats).await;
            for task in finished_tasks.iter() {
                let session_id = task.get_session_id();

                let backup_plan = {
                    let mut scheduler_guard = self.scheduler.lock().await;
                    let plan = scheduler_guard.backup(task);
                    scheduler_guard.remove_task(task);
                    plan
                };

                let (block_mapping, hash_values) = match backup_plan {
                    Some((bm, hv)) => (bm, hv),
                    None => continue,
                };

                let kv_disk_worker_group = Arc::clone(&self.kv_disk_worker_group);
                let scheduler = Arc::clone(&self.scheduler);

                let handle = tokio::spawn(async move {
                    let _ = kv_disk_worker_group.swap_out_kv(vec![block_mapping]).await;
                    scheduler
                        .lock()
                        .await
                        .release_disk_buffer(&session_id, &hash_values);
                    Ok(())
                });

                self.observer.register(handle).await;
            }
            Some(finished_tasks)
        } else {
            None
        }
    }

    fn get_local_agent_metadata(&self) -> Vec<(String, Bytes)> {
        self.local_kv_agent_metadata.clone()
    }

    async fn add_remote_agent_metadata(
        &self,
        remote_kv_agent_metadata: Vec<(String, Bytes)>,
    ) -> Result<Vec<String>> {
        let new_peer_names = self
            .kv_agent_worker_group
            .add_remote_agent_metadata(remote_kv_agent_metadata)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to set KV agent metadata: {e}"))?;

        Ok(new_peer_names)
    }

    async fn push_kv(
        &self,
        session_id: String,
        peer_names: Vec<String>,
        kv_descs: Vec<Bytes>,
        hash_values: Vec<u64>,
    ) -> Vec<u64> {
        let block_ids = self
            .scheduler
            .lock()
            .await
            .pin_buffer(&session_id, &hash_values);
        let num_blocks = block_ids.len();

        if num_blocks > 0 {
            self.kv_agent_worker_group
                .push_kv(peer_names, kv_descs, block_ids)
                .await
                .unwrap_or_else(|e| panic!("Failed to push KVs: {e}"));
        }

        self.scheduler
            .lock()
            .await
            .release_buffer(&session_id, &hash_values);

        hash_values[0..num_blocks].to_vec()
    }

    async fn clear_cache(&self) -> Result<()> {
        loop {
            {
                let mut scheduler_guard = self.scheduler.lock().await;

                if scheduler_guard.is_task_queue_empty() {
                    let _ = self.observer.cancel_all();
                    scheduler_guard.clear_cache();
                    break;
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        Ok(())
    }
}

impl Drop for LLMEngine {
    fn drop(&mut self) {
        let _ = self.observer.cancel_all();
        self.observer.shutdown();
    }
}

pub struct LLMEngineWrapper {
    engine: Arc<LLMEngine>,

    request_events: Mutex<HashMap<String, Arc<Notify>>>,
    request_outputs: Mutex<HashMap<String, InferTask>>,
}

impl LLMEngineWrapper {
    pub async fn new(
        id: String,
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        host_kv_cache_size: usize,
        disk_kv_cache_size: usize,
        disk_kv_cache_path: String,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
        worker_group_uds_path: String,
        nats_uri: String,
    ) -> Result<LLMEngineWrapper> {
        let llm_engine = LLMEngine::new(
            id,
            model_name,
            block_size,
            gpu_memory_fraction,
            host_kv_cache_size,
            disk_kv_cache_size,
            disk_kv_cache_path,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tp_size,
            worker_group_uds_path,
            nats_uri,
        )
        .await?;

        Ok(LLMEngineWrapper {
            engine: Arc::new(llm_engine),
            request_events: Mutex::new(HashMap::new()),
            request_outputs: Mutex::new(HashMap::new()),
        })
    }

    pub async fn get_id(&self) -> &str {
        &self.engine.id
    }

    pub async fn get_stats(&self) -> Result<EngineStats> {
        let stats = self.engine.get_stats().await;
        let engine_stats = EngineStats {
            num_running_reqs: stats.num_running_reqs as u64,
            num_allocated_reqs: stats.num_allocated_reqs as u64,
            num_waiting_reqs: stats.num_waiting_reqs as u64,
            num_pendding_reqs: stats.num_pendding_reqs as u64,
            gpu_kv_block_usage: stats.gpu_kv_block_usage,
            host_kv_block_usage: stats.host_kv_block_usage,
        };

        Ok(engine_stats)
    }

    pub async fn add_remote_agent(
        &self,
        remote_kv_agent_metadata: Vec<(String, Bytes)>,
    ) -> Result<()> {
        self.engine
            .add_remote_agent_metadata(remote_kv_agent_metadata)
            .await?;
        Ok(())
    }

    pub async fn generate(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
    ) -> Result<GenerateOutput> {
        let notify = Arc::new(Notify::new());
        self.request_events
            .lock()
            .await
            .insert(session_id.clone(), notify.clone());

        self.engine
            .add_request(
                input_ids,
                num_samples,
                session_id.clone(),
                max_output_len,
                ignore_eos,
            )
            .await?;

        notify.notified().await;

        let request_output: InferTask = self
            .request_outputs
            .lock()
            .await
            .remove(&session_id)
            .unwrap();

        let engine_output = GenerateOutput::from_task(&request_output);
        Ok(engine_output)
    }

    pub async fn reserve(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
    ) -> Result<ReserveOutput> {
        let (kv_descs, hash_values) = self
            .engine
            .reserve_request(
                input_ids.clone(),
                num_samples,
                session_id,
                max_output_len,
                ignore_eos,
            )
            .await?;

        Ok(ReserveOutput {
            kv_descs,
            hash_values,
        })
    }

    pub async fn transfer_kv(
        &self,
        session_id: String,
        peer_names: Vec<String>,
        kv_descs: Vec<Bytes>,
        hash_values: Vec<u64>,
    ) -> Result<TransferOutput> {
        let hash_values = self
            .engine
            .push_kv(session_id, peer_names, kv_descs, hash_values)
            .await;
        Ok(TransferOutput { hash_values })
    }

    pub async fn trigger(
        &self,
        session_id: String,
        hash_values: Vec<u64>,
    ) -> Result<GenerateOutput> {
        let notify = Arc::new(Notify::new());
        self.request_events
            .lock()
            .await
            .insert(session_id.clone(), notify.clone());

        self.engine
            .trigger_request(session_id.clone(), hash_values)
            .await?;

        notify.notified().await;

        let request_output: InferTask = self
            .request_outputs
            .lock()
            .await
            .remove(&session_id)
            .unwrap();

        let engine_output = GenerateOutput::from_task(&request_output);
        Ok(engine_output)
    }

    pub async fn get_kv_agent_metadata(&self) -> Result<Vec<(String, Bytes)>> {
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

    pub async fn clear_cache(&self) -> Result<()> {
        self.engine.clear_cache().await
    }
}
