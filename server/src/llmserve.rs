use std::collections::HashMap;
use tokenizers::tokenizer::{Result, Tokenizer};

use crate::infer_task::{InferInput, InferOutput, InferTask};
use crate::scheduler::Scheduler;
use crate::sequence::Sequence;
use crate::session_manager::{Session, SessionManager};
use crate::utils;

pub struct LLMServe {
    model_name: String,
    block_size: usize,
    gpu_memory_fraction: f32,
    max_batch_size: usize,
    max_seq_len: usize,
    max_num_batched_tokens: usize,
    tokenizer: Tokenizer,
    num_sampels: u16,

    scheduler: Scheduler,
    session_manager: SessionManager,
}

impl LLMServe {
    pub fn new(
        model_name: String,
        block_size: usize,
        gpu_memory_fraction: f32,
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
    ) -> Result<LLMServe> {
        let tokenizer = Tokenizer::from_pretrained(&model_name, None)?;
        let num_blocks: usize = 1024;

        let scheduler = Scheduler::new(
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            block_size,
            num_blocks,
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
            num_sampels: 1,
            scheduler,
            session_manager,
        })
    }

    pub fn add_request(&mut self, prompt: String, max_output_len: Option<u32>) -> Result<Vec<u32>> {
        let session = self.session_manager.create_session();

        let token_ids = self
            .tokenizer
            .encode(prompt.clone(), false)?
            .get_ids()
            .to_vec();

        let mut seqs: Vec<Sequence> = Vec::new();
        for _ in 0..self.num_sampels {
            let seq = Sequence::new(
                session.session_id.clone(),
                prompt.clone(),
                token_ids.clone(),
                max_output_len,
            );
            seqs.push(seq);
        }

        let infer_task =
            InferTask::new(session.session_id.clone(), seqs, session.last_updated_time);

        self.scheduler.add(infer_task);

        // Create session & sequences
        Ok(token_ids)
    }

    pub fn iter(&mut self) -> Option<Vec<InferTask>> {
        let infer_inputs = self.scheduler.schedule();

        // >> Model running
        let mut infer_outputs: HashMap<u64, InferOutput> = HashMap::new();
        for input in infer_inputs.iter() {
            let output_id = 0;
            infer_outputs.insert(input.seq_id, InferOutput::new(output_id));
        }

        let finished_tasks = self.scheduler.update(infer_outputs);

        if finished_tasks.len() > 0 {
            Some(finished_tasks)
        }
        else {
            None
        }
    }
}
