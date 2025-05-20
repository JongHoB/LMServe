use std::sync::Arc;

use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;
use tokenizers::tokenizer::{Result, Tokenizer};
use tokio::sync::Mutex;
use tracing::info;

use crate::infer_task::InferTask;
use crate::scheduler::Scheduler;
use crate::sequence::Sequence;
use crate::session_manager::{Session, SessionManager};
use crate::utils::{GB, generate_session_id, now};
use crate::worker::Worker;

#[allow(dead_code)]
pub struct LLMServe {
    model_name: String,
    block_size: usize,
    gpu_memory_fraction: f32,
    max_batch_size: usize,
    max_seq_len: usize,
    max_num_batched_tokens: usize,
    tokenizer: Tokenizer,

    worker_pid: u32,
    worker: Mutex<Worker>,
    scheduler: Arc<Mutex<Scheduler>>,
    session_manager: SessionManager,

    last_log_time: u64,
}

impl LLMServe {
    pub async fn new(
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
    ) -> Result<LLMServe> {
        let tokenizer = Tokenizer::from_pretrained(&model_name, None)?;

        let mut worker = Worker::new(model_name.clone(), block_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to launch llmserve-worker: {e}"));

        // After the worker warms up, return the GPU’s total memory size and
        // peak memory usage.
        let (total, peak) = worker
            .warmup(
                max_batch_size as u64,
                max_seq_len as u64,
                max_num_batched_tokens as u64,
            )
            .await
            .unwrap_or_else(|e| panic!("Failed to warmup worker: {e}"));

        let available = (total as f32 * gpu_memory_fraction) as u64;
        let cache_size = available.saturating_sub(peak);
        info!(
            "Allocating {:.2} GB of memory per GPU for KV cache.",
            cache_size as f64 / GB as f64
        );

        let num_blocks = worker
            .init_cache(cache_size)
            .await
            .unwrap_or_else(|e| panic!("Failed to init KV cache: {e}"));
        info!("Created {num_blocks} KV cache blocks.");

        let scheduler = Scheduler::new(
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            block_size,
            num_blocks as usize,
        );
        let session_manager = SessionManager::new();

        Ok(LLMServe {
            model_name,
            block_size,
            gpu_memory_fraction,
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            tokenizer,
            worker_pid: worker.get_pid(),
            worker: Mutex::new(worker),
            scheduler: Arc::new(Mutex::new(scheduler)),
            session_manager,
            last_log_time: 0,
        })
    }

    pub async fn add_request(
        &self,
        prompt: String,
        num_samples: u16,
        session_id: Option<String>,
        max_output_len: Option<usize>,
    ) -> Result<()> {
        let token_ids = self
            .tokenizer
            .encode(prompt.clone(), false)?
            .get_ids()
            .to_vec();

        let session_id: String = match session_id {
            Some(session_id) => session_id,
            None => generate_session_id(),
        };

        let mut seqs: Vec<Sequence> = Vec::new();
        for _ in 0..num_samples {
            let seq = Sequence::new(
                session_id.clone(),
                prompt.clone(),
                token_ids.clone(),
                max_output_len,
            );
            seqs.push(seq);
        }

        let infer_task = InferTask::new(session_id.clone(), seqs, now());

        {
            self.scheduler.lock().await.add(infer_task);
        }

        Ok(())
    }

    pub async fn iter(&self) -> Option<Vec<InferTask>> {
        let infer_inputs = { self.scheduler.lock().await.schedule() };
        if infer_inputs.len() == 0 {
            return None;
        }

        let infer_outputs = {
            self.worker
                .lock()
                .await
                .infer(infer_inputs)
                .await
                .unwrap_or_else(|e| panic!("Failed to execute worker: {e}"))
        };

        let finished_tasks = { self.scheduler.lock().await.update(infer_outputs) };

        if finished_tasks.len() > 0 {
            Some(finished_tasks)
        } else {
            None
        }
    }
}

impl Drop for LLMServe {
    fn drop(&mut self) {
        let result = killpg(Pid::from_raw(self.worker_pid as i32), Signal::SIGTERM);
        match result {
            Ok(_) => info!("Successfully terminated LLMServe."),
            Err(err) => info!("Failed to terminate worker: {err}"),
        }
    }
}
