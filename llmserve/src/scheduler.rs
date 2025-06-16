use std::cmp::min;
use std::collections::{HashMap, VecDeque};

use tracing::info;

use crate::block_manager::BlockManager;
use crate::infer_task::{InferInput, InferOutput, InferTask};
use crate::pb::worker::{BlockMapping, BlockMappingEntry};
use crate::sequence::SeqStatus;

struct BatchEntry {
    pub context_len: usize,
    pub chunked: bool,
}

pub struct Scheduler {
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub max_num_batched_tokens: usize,
    gpu_block_manager: BlockManager,
    pub host_block_manager: BlockManager,
    watermark_blocks: f32,
    waiting: VecDeque<InferTask>,
    allocated: VecDeque<InferTask>,

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
            running_batch: HashMap::new(),
            last_log_time: 0,
        }
    }

    pub fn add(&mut self, infer_task: InferTask) {
        for seq in infer_task.get_active_seqs() {
            self.host_block_manager.init_prefix_cache_blocks(seq);
        }

        self.waiting.push_back(infer_task);
    }

    fn try_reserve_infer_task(&mut self, infer_task: &mut InferTask, watermark: f32) -> bool {
        let seqs = infer_task.get_active_seqs_mut();

        let head_seq = seqs.first().expect("No active sequence found");

        let num_cache_blocks = self
            .gpu_block_manager
            .get_num_prefix_cache_blocks(&head_seq.token_ids);
        let num_required_blocks = self.gpu_block_manager.get_num_required_blocks(head_seq)
            - num_cache_blocks
            + seqs.len();

        if !self
            .gpu_block_manager
            .can_alloc_blocks(num_required_blocks, watermark)
        {
            return false;
        }

        for seq in seqs {
            self.gpu_block_manager.init_prefix_cache_blocks(seq);
            self.gpu_block_manager.reserve_blocks(seq);
            seq.status = SeqStatus::ALLOCATED;
        }

        true
    }

    fn try_extend_infer_task(&mut self, infer_task: &mut InferTask, watermark: f32) -> bool {
        let mut num_alloc_seqs = 0;
        let seqs = infer_task.get_active_seqs_mut();
        for seq in seqs {
            if self.gpu_block_manager.can_alloc_seq(seq, watermark) {
                self.gpu_block_manager.reserve_blocks(seq);
                seq.status = SeqStatus::ALLOCATED;
                num_alloc_seqs += 1;
            } else {
                break;
            }
        }

        num_alloc_seqs > 0
    }

    pub fn try_reserve_host_infer_task(&mut self, infer_task: &InferTask) -> bool {
        let seqs = infer_task.get_active_seqs();

        let head_seq = seqs.first().expect("No active sequence found");
        let num_required_blocks =
            self.host_block_manager.get_num_required_blocks(head_seq) + seqs.len();
        if !self
            .host_block_manager
            .can_alloc_blocks(num_required_blocks, 1.0)
        {
            return false;
        }

        for seq in seqs {
            self.host_block_manager.reserve_blocks(seq);
        }

        true
    }

    pub fn try_extend_host_infer_task(&mut self, infer_task: &InferTask) -> bool {
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

    pub fn reserve_tokens(&mut self, token_ids: &Vec<u32>, start: usize, end: usize) -> Vec<u32> {
        self.host_block_manager
            .reserve_tokens(token_ids, start, end)
    }

    fn preempt_infer_task(&mut self, infer_task: &mut InferTask) {
        let seqs = infer_task.get_active_seqs_mut();
        for seq in seqs {
            self.gpu_block_manager.free(seq.seq_id);
            seq.status = SeqStatus::WAITING;
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
        let mut write_back_block_mappings: Vec<BlockMapping> = Vec::new();

        let mut num_seqs = 0;
        let mut token_budget = self.max_num_batched_tokens;
        'outer: for infer_task in self.allocated.iter() {
            for seq in infer_task.get_seqs(SeqStatus::ALLOCATED) {
                if num_seqs >= self.max_batch_size || token_budget == 0 {
                    break 'outer;
                }

                let total = seq.token_ids.len();
                let mut filled = self.gpu_block_manager.get_filled_token_len(seq.seq_id);
                {
                    let gpu_filled = self.gpu_block_manager.get_filled_token_len(seq.seq_id);
                    let host_filled = self.host_block_manager.get_filled_token_len(seq.seq_id);

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
                }

                // Although all tokens are filled, we use last token to generate an output token.
                filled = min(filled, total - 1);
                let input_len = min(total - filled, token_budget);

                if input_len > 0 {
                    let input_ids = seq.token_ids[filled..filled + input_len].to_vec();
                    let (block_ids, _) = self.gpu_block_manager.get_block_ids_range(
                        seq.seq_id,
                        0,
                        filled + input_len,
                    );

                    let chunked = total > (filled + input_len);

                    let input_len = input_ids.len();
                    let context_len = input_len + filled;

                    let infer_input =
                        InferInput::new(seq.seq_id, input_ids, filled, context_len, block_ids);

                    let entry = BatchEntry {
                        context_len,
                        chunked,
                    };

                    infer_inputs.push(infer_input);

                    num_seqs += 1;
                    token_budget -= input_len;

                    running_batch.insert(seq.seq_id, entry);

                    {
                        let host_filled = self.host_block_manager.get_filled_token_len(seq.seq_id);

                        let (gpu_block_ids, _) = self.gpu_block_manager.get_block_ids_range(
                            seq.seq_id,
                            host_filled,
                            context_len,
                        );
                        let (host_block_ids, _) = self.host_block_manager.get_block_ids_range(
                            seq.seq_id,
                            host_filled,
                            context_len,
                        );

                        let mut block_entries: Vec<_> = Vec::with_capacity(gpu_block_ids.len());
                        for (gpu_blk_id, host_blk_id) in
                            gpu_block_ids.into_iter().zip(host_block_ids)
                        {
                            block_entries.push(BlockMappingEntry {
                                src_block_id: gpu_blk_id,
                                dst_block_id: host_blk_id,
                            });
                        }

                        write_back_block_mappings.push(BlockMapping {
                            entries: block_entries,
                        });
                    }
                }
            }
        }

        (
            running_batch,
            infer_inputs,
            fetch_block_mappings,
            write_back_block_mappings,
        )
    }

    pub fn schedule(&mut self) -> (Vec<InferInput>, Vec<BlockMapping>, Vec<BlockMapping>) {
        let mut allocated: VecDeque<InferTask> = VecDeque::new();

        while let Some(mut infer_task) = self.allocated.pop_front() {
            if self.try_extend_infer_task(&mut infer_task, 1.0) {
                for seq in infer_task.get_active_seqs_mut() {
                    seq.status = SeqStatus::ALLOCATED;
                }
                self.try_extend_host_infer_task(&infer_task);

                allocated.push_back(infer_task);
            } else {
                if let Some(mut preempt_task) = self.allocated.pop_back() {
                    self.preempt_infer_task(&mut preempt_task);
                    self.waiting.push_front(preempt_task);

                    self.allocated.push_front(infer_task);
                    continue;
                } else {
                    self.waiting.push_front(infer_task);
                    break;
                }
            }
        }

        while let Some(mut infer_task) = self.waiting.pop_front() {
            if self.try_reserve_infer_task(&mut infer_task, self.watermark_blocks) {
                for seq in infer_task.get_active_seqs_mut() {
                    seq.status = SeqStatus::ALLOCATED;
                }
                self.try_reserve_host_infer_task(&infer_task);

                allocated.push_back(infer_task);
            } else {
                self.waiting.push_front(infer_task);
                break;
            }
        }

        self.allocated = allocated;

        let (running_batch, infer_inputs, fetch_block_mappings, write_back_block_mappings) =
            self.dispatch();

        self.running_batch = running_batch;

        {
            // Logging
            let now = utils::time::now_ns();

            if now > self.last_log_time + 5e9 as u64 {
                info!(
                    "Running: {} seqs, {:}",
                    infer_inputs.len(),
                    self.get_log_text()
                );
                self.last_log_time = now;
            }
        }

        (
            infer_inputs,
            fetch_block_mappings,
            write_back_block_mappings,
        )
    }

    pub fn commit(&mut self, infer_outputs: HashMap<u64, InferOutput>) -> Vec<InferTask> {
        let mut finished_tasks: Vec<InferTask> = Vec::new();
        let mut still_allocated_tasks = VecDeque::with_capacity(self.allocated.len());
        let now = utils::time::now_ns();

        for mut task in self.allocated.drain(..) {
            for seq in task.get_seqs_mut(SeqStatus::ALLOCATED) {
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
                    .update_filled_len(seq.seq_id, entry.context_len);

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

                let reached_max_len = seq
                    .max_output_len
                    .map_or(false, |max| seq.output_len >= max);

                if (!seq.ignore_eos && output.is_eos) || reached_max_len {
                    seq.status = SeqStatus::FINISHED;
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

    pub fn get_host_cache_block_range(
        &self,
        token_ids: Vec<u32>,
        start: usize,
        end: usize,
    ) -> (Vec<u32>, usize) {
        self.host_block_manager
            .get_prefix_cache_blocks_range(&token_ids, start, end)
    }

    pub fn get_host_cache_token_len(&self, token_ids: &Vec<u32>) -> usize {
        self.host_block_manager
            .get_prefix_cache_token_len(token_ids)
    }

    fn remove_task(&mut self, infer_task: &InferTask) {
        for seq in infer_task.get_seqs(SeqStatus::FINISHED) {
            self.gpu_block_manager.free(seq.seq_id);
            self.host_block_manager.free(seq.seq_id);
        }
    }

    fn get_log_text(&self) -> String {
        let num_allocated_reqs: usize = self.allocated.len();
        let num_waiting_reqs: usize = self.waiting.len();
        let kv_block_usage: f32 = self.gpu_block_manager.get_block_usage();

        format!(
            "Allocated: {} reqs, Waiting: {} reqs, KV cache usage: {:.2} %",
            num_allocated_reqs,
            num_waiting_reqs,
            kv_block_usage * 100.0
        )
    }
}
