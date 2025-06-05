use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use tokio::join;
use tokio::sync::{Mutex, Notify};
use tracing::info;

use crate::infer_task::InferTask;
use crate::scheduler::Scheduler;
use crate::sequence::{SeqStatus, Sequence};
use crate::worker::{KVWorkerGroup, ModelWorkerGroup};

pub fn norm_log_probs(probs: &[f32]) -> f32 {
    let log_sum: f32 = probs
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
        .sum();

    log_sum / probs.len() as f32
}

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

    scheduler: Arc<Mutex<Scheduler>>,

    last_log_time: u64,
}

impl LLMEngine {
    pub async fn new(
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
    ) -> Result<LLMEngine> {
        let model_worker_group = ModelWorkerGroup::init(tp_size)
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

        let num_blocks = model_worker_group
            .init_cache(cache_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to init KV cache: {e}"));
        info!("Created {num_blocks} KV cache blocks.");

        let kv_worker_group = KVWorkerGroup::init(tp_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to initialize KV worker group: {e}"));

        let scheduler = Scheduler::new(
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            block_size,
            num_blocks as usize,
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
            scheduler: Arc::new(Mutex::new(scheduler)),
            last_log_time: 0,
        })
    }

    pub async fn add_request(
        &self,
        input_ids: Vec<u32>,
        num_samples: u16,
        session_id: String,
        max_output_len: Option<usize>,
        ignore_eos: bool,
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

        let infer_task = InferTask::new(session_id.clone(), seqs, utils::time::now());

        {
            self.scheduler.lock().await.add(infer_task);
        }

        Ok(())
    }

    pub async fn iter(&self) -> Option<Vec<InferTask>> {
        let (infer_inputs, block_mappings) = { self.scheduler.lock().await.schedule() };
        if infer_inputs.len() == 0 {
            return None;
        }

        let record_after_execute = block_mappings.len() > 0;

        let (infer_result, transfer_result) = join!(
            self.model_worker_group
                .infer(infer_inputs, false, record_after_execute),
            self.kv_worker_group.transfer_kv(block_mappings),
        );

        let infer_outputs =
            infer_result.unwrap_or_else(|e| panic!("Failed to execute worker: {e}"));
        let _ = transfer_result.unwrap_or_else(|e| panic!("Failed to transfer KV: {e}"));

        let finished_tasks = { self.scheduler.lock().await.update(infer_outputs) };

        if finished_tasks.len() > 0 {
            Some(finished_tasks)
        } else {
            None
        }
    }
}

pub struct LLMEngineOutput {
    pub session_id: String,
    pub output_ids: Vec<u32>,
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
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        tp_size: u8,
    ) -> Result<LLMEngineWrapper> {
        let llm_engine = LLMEngine::new(
            model_name,
            block_size,
            gpu_memory_fraction,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tp_size,
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
            )
            .await?;

        notify.notified().await;

        let request_output = {
            self.request_outputs
                .lock()
                .await
                .remove(&session_id)
                .unwrap()
        };
        let seqs = request_output.get_seqs(SeqStatus::FINISHED);
        let selected_seq = seqs
            .iter()
            .max_by(|a, b| {
                norm_log_probs(a.get_token_probs().as_ref())
                    .partial_cmp(&norm_log_probs(b.get_token_probs().as_ref()))
                    .unwrap()
            })
            .expect("Failed to select sequence: max_by() returned None.");

        let output_ids = selected_seq.get_output_ids();

        Ok(LLMEngineOutput {
            session_id: session_id.clone(),
            output_ids: output_ids.to_vec(),
        })
    }

    pub async fn run_engine(&self) -> Result<()> {
        loop {
            let infer_tasks = self.engine.iter().await;
            let Some(infer_tasks) = infer_tasks else {
                continue;
            };
            for infer_task in infer_tasks.into_iter() {
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
