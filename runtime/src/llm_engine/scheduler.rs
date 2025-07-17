use std::collections::{HashMap, VecDeque};
use std::fmt;

use tracing::{debug, info};

use crate::pb::worker::{BlockMapping, BlockMappingEntry};

use super::block_manager::BlockManager;
use super::infer_task::{InferInput, InferOutput, InferTask};
use super::sequence::SeqStatus;
use super::sequence::Sequence;

pub struct SchedStatus {
    pub num_running_reqs: usize,
    pub num_allocated_reqs: usize,
    pub num_waiting_reqs: usize,
    pub num_pendding_reqs: usize,
    pub gpu_kv_block_usage: f32,
    pub host_kv_block_usage: f32,
}

impl fmt::Display for SchedStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Running: {} seqs, Allocated: {} reqs, Waiting: {} reqs, Pending: {} reqs, GPU KV usage: {:.2} %, Host KV usage: {:.2} %",
            self.num_running_reqs,
            self.num_allocated_reqs,
            self.num_waiting_reqs,
            self.num_pendding_reqs,
            self.gpu_kv_block_usage * 100.0,
            self.host_kv_block_usage * 100.0,
        )
    }
}

struct BatchEntry {
    pub context_len: usize,
    pub chunked: bool,
    pub caching_context_len: usize,
}

pub struct Scheduler {
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub max_num_batched_tokens: usize,
    gpu_block_manager: BlockManager,
    host_block_manager: BlockManager,
    watermark_blocks: f32,
    waiting: VecDeque<InferTask>,
    allocated: VecDeque<InferTask>,
    // HashMap<session_id>, infer_task>
    pendding: HashMap<String, InferTask>,

    running_batch: HashMap<u64, BatchEntry>,

    last_log_time: u64,
}

impl Scheduler {
    pub fn new(
        max_batch_size: usize,
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        block_size: usize,
        num_gpu_blocks: usize,
        num_host_blocks: usize,
    ) -> Scheduler {
        Scheduler {
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            gpu_block_manager: BlockManager::new(block_size, num_gpu_blocks),
            host_block_manager: BlockManager::new(block_size, num_host_blocks),
            watermark_blocks: 0.97,
            waiting: VecDeque::new(),
            allocated: VecDeque::new(),
            pendding: HashMap::new(),
            running_batch: HashMap::new(),
            last_log_time: 0,
        }
    }

    pub fn is_task_queue_empty(&self) -> bool {
        self.allocated.is_empty() && self.waiting.is_empty() && self.pendding.is_empty()
    }

    pub fn add(&mut self, infer_task: InferTask) {
        let seqs = infer_task.get_active_seqs();
        for seq in seqs.iter() {
            self.host_block_manager.init_prefix_cache_blocks(seq);
        }

        self.waiting.push_back(infer_task);
    }

    pub fn pend(&mut self, infer_task: InferTask) {
        let old = self
            .pendding
            .insert(infer_task.get_session_id(), infer_task);
        if old.is_some() {
            panic!("A duplicate session id is already pending");
        }
    }

    pub fn trigger_pend_task(&mut self, session_id: String, hash_values: &[u64]) {
        let infer_task = self
            .pendding
            .remove(&session_id)
            .unwrap_or_else(|| panic!("no pending task found for session_id: {}", session_id));

        if !hash_values.is_empty() {
            let elapsed_time = utils::time::now_ns() - infer_task.get_arrival_time();
            debug!(
                "Transferred {} blocks in {} ms",
                hash_values.len(),
                elapsed_time / 1_000_000
            );
        }

        self.release_buffer(&session_id, hash_values);

        let head_seq = infer_task.get_head_seq().expect("No active sequence found");
        self.host_block_manager.update_prefix_cache_blocks(head_seq);

        self.add(infer_task);
    }

    fn try_reserve_infer_task(&mut self, infer_task: &mut InferTask, watermark: f32) -> bool {
        let mut num_alloc_seqs = 0;
        for seq in infer_task.get_active_seqs_mut() {
            let num_cache_blocks = self
                .gpu_block_manager
                .get_num_prefix_cache_blocks(&seq.token_ids);
            let num_required_blocks =
                self.gpu_block_manager.get_num_required_blocks(seq) - num_cache_blocks;
            if self
                .gpu_block_manager
                .can_alloc_blocks(num_required_blocks, watermark)
            {
                self.gpu_block_manager.init_prefix_cache_blocks(seq);

                self.gpu_block_manager.reserve_blocks(seq);
                seq.status = SeqStatus::Allocated;
                num_alloc_seqs += 1;
            } else {
                break;
            }
        }

        num_alloc_seqs > 0
    }

    fn try_extend_infer_task(&mut self, infer_task: &mut InferTask, watermark: f32) -> bool {
        let mut num_alloc_seqs = 0;
        let seqs = infer_task.get_active_seqs_mut();
        for seq in seqs {
            if self.gpu_block_manager.can_alloc_seq(seq, watermark) {
                self.gpu_block_manager.reserve_blocks(seq);
                seq.status = SeqStatus::Allocated;
                num_alloc_seqs += 1;
            } else {
                break;
            }
        }

        num_alloc_seqs > 0
    }

    fn try_extend_host_infer_task(&mut self, infer_task: &InferTask) -> bool {
        let mut num_alloc_seqs = 0;
        let seqs = infer_task.get_active_seqs();
        for seq in seqs {
            if self.host_block_manager.can_alloc_seq(seq, 1.0) {
                self.host_block_manager.reserve_blocks(seq);
                num_alloc_seqs += 1;
            } else {
                break;
            }
        }

        num_alloc_seqs > 0
    }

    pub fn init_prefix_host_cache_blocks(&mut self, seq: &Sequence) -> usize {
        self.host_block_manager.init_prefix_cache_blocks(seq)
    }

    pub fn reserve_buffer(
        &mut self,
        session_id: &str,
        token_ids: &[u32],
        start: usize,
    ) -> (Vec<u32>, Vec<u64>) {
        self.host_block_manager
            .reserve_buffer_by_tokens(session_id, token_ids, start)
    }

    pub fn pin_buffer(&mut self, session_id: &str, hash_values: &[u64]) -> Vec<u32> {
        self.host_block_manager
            .pin_buffer_by_hashes(session_id, hash_values)
    }

    pub fn release_buffer(&mut self, session_id: &str, hash_values: &[u64]) {
        self.host_block_manager
            .release_buffer(session_id, hash_values);
    }

    fn preempt_infer_task(&mut self, infer_task: &mut InferTask) {
        let seqs = infer_task.get_active_seqs_mut();
        for seq in seqs {
            self.gpu_block_manager.free(seq.seq_id);
            seq.status = SeqStatus::Waiting;
        }
    }

    fn dispatch(
        &mut self,
    ) -> (
        HashMap<u64, BatchEntry>,
        Vec<InferInput>,
        Vec<BlockMapping>,
        Vec<BlockMapping>,
    ) {
        let mut running_batch: HashMap<u64, BatchEntry> = HashMap::new();
        let mut infer_inputs: Vec<InferInput> = Vec::new();
        let mut fetch_block_mappings: Vec<BlockMapping> = Vec::new();
        let mut write_through_block_mappings: Vec<BlockMapping> = Vec::new();

        let mut token_budget = self.max_num_batched_tokens;
        'outer: for infer_task in self.allocated.iter() {
            for seq in infer_task.get_seqs(SeqStatus::Allocated) {
                if running_batch.len() >= self.max_batch_size || token_budget == 0 {
                    break 'outer;
                }

                let total = seq.token_ids.len();
                let filled = {
                    let gpu_filled = self.gpu_block_manager.get_filled_token_len(seq.seq_id);
                    let host_filled = self.host_block_manager.get_filled_token_len(seq.seq_id);

                    // If the cached tokens in host memory has more than GPU,
                    // it makes a block mapping to fetch the remaiing tokens.
                    if host_filled > gpu_filled {
                        let (gpu_block_ids, _) = self.gpu_block_manager.get_block_ids_range(
                            seq.seq_id,
                            gpu_filled,
                            host_filled,
                        );
                        let (host_block_ids, _) = self.host_block_manager.get_block_ids_range(
                            seq.seq_id,
                            gpu_filled,
                            host_filled,
                        );

                        let mut block_entries: Vec<_> =
                            Vec::with_capacity(host_block_ids.len().min(gpu_block_ids.len()));
                        for (host_blk_id, gpu_blk_id) in
                            host_block_ids.into_iter().zip(gpu_block_ids)
                        {
                            block_entries.push(BlockMappingEntry {
                                src_block_id: host_blk_id,
                                dst_block_id: gpu_blk_id,
                            });
                        }

                        fetch_block_mappings.push(BlockMapping {
                            entries: block_entries,
                        });

                        self.gpu_block_manager
                            .update_filled_len(seq.seq_id, host_filled);
                    }

                    // Although all tokens are filled, we use last token to generate an output token.
                    gpu_filled.max(host_filled).min(total.saturating_sub(1))
                };

                let input_len = total.saturating_sub(filled).min(token_budget);

                if input_len > 0 {
                    let input_ids = seq.token_ids[filled..filled + input_len].to_vec();
                    let (block_ids, _) = self.gpu_block_manager.get_block_ids_range(
                        seq.seq_id,
                        0,
                        filled + input_len,
                    );

                    let input_len = input_ids.len();
                    let context_len = input_len + filled;

                    let infer_input =
                        InferInput::new(seq.seq_id, input_ids, filled, context_len, block_ids);

                    infer_inputs.push(infer_input);
                    token_budget -= input_len;

                    let caching_context_len = {
                        let host_filled = self.host_block_manager.get_filled_token_len(seq.seq_id);
                        let num_host_allocated_slots =
                            self.host_block_manager.get_num_allocated_slots(seq.seq_id);

                        let (gpu_block_ids, _) = self.gpu_block_manager.get_block_ids_range(
                            seq.seq_id,
                            host_filled,
                            context_len,
                        );
                        let (host_block_ids, _) = self.host_block_manager.get_block_ids_range(
                            seq.seq_id,
                            host_filled,
                            num_host_allocated_slots,
                        );

                        //  Newly generated tokens on the GPU are immediately written through to host memory
                        let mut block_entries: Vec<_> =
                            Vec::with_capacity(gpu_block_ids.len().min(host_block_ids.len()));
                        for (gpu_blk_id, host_blk_id) in
                            gpu_block_ids.into_iter().zip(host_block_ids)
                        {
                            block_entries.push(BlockMappingEntry {
                                src_block_id: gpu_blk_id,
                                dst_block_id: host_blk_id,
                            });
                        }

                        write_through_block_mappings.push(BlockMapping {
                            entries: block_entries,
                        });

                        context_len.min(num_host_allocated_slots)
                    };

                    // The entry is required to update the scheduling states
                    let entry = BatchEntry {
                        context_len,
                        chunked: total > (filled + input_len),
                        caching_context_len,
                    };

                    running_batch.insert(seq.seq_id, entry);
                }
            }
        }

        (
            running_batch,
            infer_inputs,
            fetch_block_mappings,
            write_through_block_mappings,
        )
    }

    pub fn schedule(&mut self) -> (Vec<InferInput>, Vec<BlockMapping>, Vec<BlockMapping>) {
        let mut allocated: VecDeque<InferTask> = VecDeque::new();

        while let Some(mut infer_task) = self.allocated.pop_front() {
            if self.try_extend_infer_task(&mut infer_task, 1.0) {
                for seq in infer_task.get_active_seqs_mut() {
                    seq.status = SeqStatus::Allocated;
                }
                self.try_extend_host_infer_task(&infer_task);

                allocated.push_back(infer_task);
            } else if let Some(mut preempt_task) = self.allocated.pop_back() {
                self.preempt_infer_task(&mut preempt_task);
                self.waiting.push_front(preempt_task);

                self.allocated.push_front(infer_task);
                continue;
            } else {
                self.waiting.push_front(infer_task);
                break;
            }
        }

        while let Some(mut infer_task) = self.waiting.pop_front() {
            if self.try_reserve_infer_task(&mut infer_task, self.watermark_blocks) {
                for seq in infer_task.get_active_seqs_mut() {
                    seq.status = SeqStatus::Allocated;
                }
                self.try_extend_host_infer_task(&infer_task);

                allocated.push_back(infer_task);
            } else {
                self.waiting.push_front(infer_task);
                break;
            }
        }

        self.allocated = allocated;

        let (running_batch, infer_inputs, fetch_block_mappings, write_through_block_mappings) =
            self.dispatch();

        self.running_batch = running_batch;

        {
            // Logging
            let now = utils::time::now_ns();

            if now > self.last_log_time + 5e9 as u64 {
                info!("{:}", self.get_status().to_string());
                self.last_log_time = now;
            }
        }

        (
            infer_inputs,
            fetch_block_mappings,
            write_through_block_mappings,
        )
    }

    pub fn commit(&mut self, infer_outputs: HashMap<u64, InferOutput>) -> Vec<InferTask> {
        let mut finished_tasks: Vec<InferTask> = Vec::new();
        let mut still_allocated_tasks = VecDeque::with_capacity(self.allocated.len());
        let now = utils::time::now_ns();

        for mut task in self.allocated.drain(..) {
            for seq in task.get_seqs_mut(SeqStatus::Allocated) {
                let (output, entry) = match (
                    infer_outputs.get(&seq.seq_id),
                    self.running_batch.get(&seq.seq_id),
                ) {
                    (Some(output), Some(entry)) => (output, entry),
                    _ => continue,
                };

                self.gpu_block_manager
                    .update_filled_len(seq.seq_id, entry.context_len);
                self.host_block_manager
                    .update_filled_len(seq.seq_id, entry.caching_context_len);

                if entry.chunked {
                    continue;
                }

                seq.cached = false;
                seq.append_output_id(
                    output.output_id,
                    output.prob,
                    output.output_word.clone(),
                    now,
                );

                let reached_max_len = seq.max_output_len.is_some_and(|max| seq.output_len >= max)
                    || seq.token_ids.len() >= self.max_seq_len;

                if (!seq.ignore_eos && output.is_eos) || reached_max_len {
                    seq.status = SeqStatus::Finished;
                }
            }

            if task.is_finished() {
                finished_tasks.push(task);
            } else {
                still_allocated_tasks.push_back(task);
            }
        }

        for task in finished_tasks.iter() {
            self.remove_task(task);
        }

        self.allocated = still_allocated_tasks;
        self.running_batch.clear();

        finished_tasks
    }

    fn remove_task(&mut self, infer_task: &InferTask) {
        for seq in infer_task.get_seqs(SeqStatus::Finished) {
            self.gpu_block_manager.free(seq.seq_id);
            self.host_block_manager.free(seq.seq_id);
        }
    }

    pub fn clear_cache(&mut self) {
        self.gpu_block_manager.clear_cache();
        self.host_block_manager.clear_cache();
    }

    pub fn get_status(&self) -> SchedStatus {
        let num_running_reqs: usize = self.running_batch.len();
        let num_allocated_reqs: usize = self.allocated.len();
        let num_waiting_reqs: usize = self.waiting.len();
        let num_pendding_reqs: usize = self.pendding.len();
        let gpu_kv_block_usage: f32 = self.gpu_block_manager.get_block_usage();
        let host_kv_block_usage: f32 = self.host_block_manager.get_block_usage();

        SchedStatus {
            num_running_reqs,
            num_allocated_reqs,
            num_waiting_reqs,
            num_pendding_reqs,
            gpu_kv_block_usage,
            host_kv_block_usage,
        }
    }
}
