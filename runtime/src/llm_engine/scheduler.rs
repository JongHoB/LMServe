use std::collections::{HashMap, VecDeque};

use tracing::info;

use crate::pb::worker::{BlockMapping, BlockMappingEntry};
use crate::stats::Stats;

use super::block_manager::{BlockManager, BlockRegion};
use super::infer_task::{InferInput, InferOutput, InferTask};
use super::sequence::SeqStatus;
use super::sequence::Sequence;

const DISK_RECOMPUTE_THRESHOLD: usize = 2048;

struct BatchEntry {
    pub chunked: bool,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum Device {
    Gpu,
    Host,
    Disk,
}

#[derive(Debug, Clone)]
pub struct DeviceBlockRegion {
    device: Device,
    block_region: BlockRegion,
}

impl DeviceBlockRegion {
    pub fn region_id(&self) -> &str {
        self.block_region.id()
    }

    pub fn block_ids(&self) -> &[u32] {
        self.block_region.block_ids()
    }

    pub fn hashes(&self) -> &[u64] {
        self.block_region.hashes()
    }
}

pub struct Scheduler {
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub max_num_batched_tokens: usize,
    gpu_block_manager: BlockManager,
    host_block_manager: BlockManager,
    disk_block_manager: BlockManager,
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
        num_disk_blocks: usize,
    ) -> Scheduler {
        Scheduler {
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
            gpu_block_manager: BlockManager::new(block_size, num_gpu_blocks),
            host_block_manager: BlockManager::new(block_size, num_host_blocks),
            disk_block_manager: BlockManager::new(block_size, num_disk_blocks),
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
        self.waiting.push_back(infer_task);
    }

    fn get_block_manager(&self, device: &Device) -> &BlockManager {
        match device {
            Device::Gpu => &self.gpu_block_manager,
            Device::Host => &self.host_block_manager,
            Device::Disk => &self.disk_block_manager,
        }
    }

    fn get_block_manager_mut(&mut self, device: &Device) -> &mut BlockManager {
        match device {
            Device::Gpu => &mut self.gpu_block_manager,
            Device::Host => &mut self.host_block_manager,
            Device::Disk => &mut self.disk_block_manager,
        }
    }

    pub fn plan_stage(
        &mut self,
        infer_task: &InferTask,
    ) -> Option<(DeviceBlockRegion, BlockMapping)> {
        self.init_prefix_cache_blocks(infer_task, &[Device::Host, Device::Disk]);

        let head_seq = infer_task.get_head_seq().expect("No active sequence found");

        let host_filled = self
            .host_block_manager
            .get_filled_token_len(head_seq.seq_id);
        let disk_filled = self
            .disk_block_manager
            .get_filled_token_len(head_seq.seq_id);
        // NOTE(jinu):
        // If no pending requests and the number of reusable KVs on disk is below threshold,
        // recompute instead of waiting for KV restoration.
        if self.waiting.is_empty() && (disk_filled - host_filled) < DISK_RECOMPUTE_THRESHOLD {
            return None;
        }

        self.reserve_copy_blocks(head_seq, Device::Disk, Device::Host)
    }

    pub fn backup(&mut self, infer_task: &InferTask) -> Option<(DeviceBlockRegion, BlockMapping)> {
        let finished_seqs = infer_task.get_seqs(SeqStatus::Finished);
        let head_finished_seq = finished_seqs.first().expect("No finished sequence found");

        self.reserve_copy_blocks(head_finished_seq, Device::Host, Device::Disk)
    }

    pub fn pend(&mut self, infer_task: InferTask) {
        let old = self
            .pendding
            .insert(infer_task.get_session_id(), infer_task);
        if old.is_some() {
            panic!("A duplicate session id is already pending");
        }
    }

    pub fn trigger_pend_task(
        &mut self,
        session_id: String,
        region_id: Option<String>,
        hash_values: &[u64],
    ) {
        let infer_task = self
            .pendding
            .remove(&session_id)
            .unwrap_or_else(|| panic!("no pending task found for session_id: {}", session_id));

        if let Some(region_id) = region_id {
            self.host_block_manager
                .activate_reserved_blocks(&region_id, hash_values);
        }

        let head_seq = infer_task.get_head_seq().expect("No active sequence found");
        self.host_block_manager.update_prefix_cache_blocks(head_seq);

        self.add(infer_task);
    }

    fn try_alloc_infer_task(
        &mut self,
        infer_task: &mut InferTask,
        watermark: f32,
        device: Device,
    ) -> bool {
        let mut num_alloc_seqs = 0;
        let block_manager = self.get_block_manager_mut(&device);
        for seq in infer_task.get_active_seqs_mut() {
            let num_cache_blocks = block_manager.get_num_prefix_cache_blocks(&seq.token_ids);
            let num_required_blocks = block_manager.get_num_required_blocks(seq) - num_cache_blocks;
            if block_manager.can_alloc_blocks(num_required_blocks, watermark) {
                block_manager.init_prefix_cache_blocks(seq);

                block_manager.alloc_blocks(seq);
                seq.status = SeqStatus::Allocated;
                num_alloc_seqs += 1;
            } else {
                break;
            }
        }

        num_alloc_seqs > 0
    }

    fn try_extend_infer_task(
        &mut self,
        infer_task: &InferTask,
        watermark: f32,
        device: Device,
    ) -> bool {
        let mut num_alloc_seqs = 0;
        let seqs = infer_task.get_active_seqs();

        let block_manager = self.get_block_manager_mut(&device);

        for seq in seqs {
            if block_manager.can_alloc_seq(seq, watermark) {
                block_manager.alloc_blocks(seq);
                num_alloc_seqs += 1;
            } else {
                break;
            }
        }

        num_alloc_seqs > 0
    }

    pub fn init_prefix_cache_blocks(&mut self, infer_task: &InferTask, devices: &[Device]) {
        for seq in infer_task.get_active_seqs() {
            for device in devices {
                self.get_block_manager_mut(device)
                    .init_prefix_cache_blocks(seq);
            }
        }
    }

    pub fn reserve_blocks(
        &mut self,
        infer_task: &InferTask,
        device: Device,
    ) -> Option<DeviceBlockRegion> {
        let head_seq = infer_task.get_head_seq().expect("No active sequence found");

        let block_manager = self.get_block_manager_mut(&device);
        let filled = block_manager.get_filled_token_len(head_seq.seq_id);

        block_manager
            .reserve_blocks(head_seq.get_token_ids(), filled)
            .map(|region| DeviceBlockRegion {
                device,
                block_region: region,
            })
    }

    pub fn commit_reserved_blocks(&mut self, block_region: DeviceBlockRegion) -> usize {
        self.get_block_manager_mut(&block_region.device)
            .activate_reserved_blocks(block_region.region_id(), block_region.hashes())
    }

    pub fn pin_blocks(&mut self, hash_values: &[u64], device: Device) -> DeviceBlockRegion {
        let block_region = self
            .get_block_manager_mut(&device)
            .pin_blocks_by_hashes(hash_values);

        DeviceBlockRegion {
            device,
            block_region,
        }
    }

    pub fn unpin_blocks(&mut self, block_region: DeviceBlockRegion) {
        self.get_block_manager_mut(&block_region.device)
            .unpin_blocks(block_region.region_id());
    }

    fn preempt_infer_task(&mut self, infer_task: &mut InferTask) {
        let seqs = infer_task.get_active_seqs_mut();
        for seq in seqs {
            self.gpu_block_manager.free(seq.seq_id);
            seq.status = SeqStatus::Waiting;
        }
    }

    fn reserve_copy_blocks(
        &mut self,
        seq: &Sequence,
        src_dev: Device,
        dst_dev: Device,
    ) -> Option<(DeviceBlockRegion, BlockMapping)> {
        let src_block_manager = self.get_block_manager(&src_dev);
        let dst_block_manager = self.get_block_manager(&dst_dev);

        let src_filled = src_block_manager.get_filled_token_len(seq.seq_id);
        let dst_filled = dst_block_manager.get_filled_token_len(seq.seq_id);

        if src_filled <= dst_filled {
            return None;
        }

        let (src_block_ids, _) =
            src_block_manager.get_block_ids_range(seq.seq_id, dst_filled, src_filled, false)?;

        let num_required_blocks = src_block_ids.len();
        if !dst_block_manager.can_alloc_blocks(num_required_blocks, 1.0) {
            return None;
        }

        self.get_block_manager_mut(&dst_dev)
            .reserve_blocks(&seq.get_token_ids()[..src_filled], dst_filled)
            .map(|region| {
                let block_entries: Vec<BlockMappingEntry> = src_block_ids
                    .iter()
                    .zip(region.block_ids())
                    .map(|(&sid, &did)| BlockMappingEntry {
                        src_block_id: sid,
                        dst_block_id: did,
                    })
                    .collect();

                (
                    DeviceBlockRegion {
                        device: dst_dev,
                        block_region: region,
                    },
                    BlockMapping {
                        entries: block_entries,
                    },
                )
            })
    }

    fn copy_blocks(
        &self,
        seq: &Sequence,
        src_dev: Device,
        dst_dev: Device,
        include_partial_block: bool,
    ) -> Option<(BlockMapping, usize)> {
        let src_block_manager = self.get_block_manager(&src_dev);
        let dst_block_manager = self.get_block_manager(&dst_dev);

        let src_filled = src_block_manager.get_filled_token_len(seq.seq_id);
        let dst_filled = dst_block_manager.get_filled_token_len(seq.seq_id);

        let num_dst_allocated_slots = dst_block_manager.get_num_allocated_slots(seq.seq_id);
        let end = src_filled.min(num_dst_allocated_slots);

        // If the cached tokens in src device has more than dst device,
        // it makes a block mapping to fetch the remaining tokens.
        if end > dst_filled {
            let (src_block_ids, src_last_filled_token_idx) = src_block_manager
                .get_block_ids_range(seq.seq_id, dst_filled, end, include_partial_block)?;
            let (dst_block_ids, _) = dst_block_manager.get_block_ids_range(
                seq.seq_id,
                dst_filled,
                end,
                include_partial_block,
            )?;

            let mut block_entries: Vec<_> =
                Vec::with_capacity(src_block_ids.len().min(dst_block_ids.len()));
            for (src_blk_id, dst_blk_id) in src_block_ids.into_iter().zip(dst_block_ids) {
                block_entries.push(BlockMappingEntry {
                    src_block_id: src_blk_id,
                    dst_block_id: dst_blk_id,
                });
            }

            Some((
                BlockMapping {
                    entries: block_entries,
                },
                src_last_filled_token_idx,
            ))
        } else {
            None
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

                if let Some((block_mapping, filled_end)) =
                    self.copy_blocks(seq, Device::Host, Device::Gpu, true)
                {
                    fetch_block_mappings.push(block_mapping);
                    self.gpu_block_manager
                        .update_filled_len(seq.seq_id, filled_end);
                }

                // Although all tokens are filled, we use last token to generate an output token.
                let filled = self
                    .gpu_block_manager
                    .get_filled_token_len(seq.seq_id)
                    .min(total.saturating_sub(1));

                let input_len = total.saturating_sub(filled).min(token_budget);

                if input_len > 0 {
                    let input_ids = seq.token_ids[filled..filled + input_len].to_vec();
                    let block_range = self.gpu_block_manager.get_block_ids_range(
                        seq.seq_id,
                        0,
                        filled + input_len,
                        true,
                    );

                    let block_ids = match block_range {
                        Some((block_ids, _)) => block_ids,
                        None => continue,
                    };

                    let input_len = input_ids.len();
                    let context_len = input_len + filled;

                    let infer_input =
                        InferInput::new(seq.seq_id, input_ids, filled, context_len, block_ids);

                    infer_inputs.push(infer_input);
                    token_budget -= input_len;

                    self.gpu_block_manager
                        .update_filled_len(seq.seq_id, context_len);

                    if let Some((block_mapping, filled_end)) =
                        self.copy_blocks(seq, Device::Gpu, Device::Host, false)
                    {
                        write_through_block_mappings.push(block_mapping);
                        self.host_block_manager
                            .update_filled_len(seq.seq_id, filled_end);
                    }

                    // The entry is required to update the scheduling states
                    let entry = BatchEntry {
                        chunked: total > (filled + input_len),
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

    pub fn schedule(&mut self) -> (Vec<InferInput>, Vec<BlockMapping>, Vec<BlockMapping>, Stats) {
        let mut allocated: VecDeque<InferTask> = VecDeque::new();

        while let Some(infer_task) = self.allocated.pop_front() {
            if self.try_extend_infer_task(&infer_task, 1.0, Device::Gpu) {
                self.try_extend_infer_task(&infer_task, 1.0, Device::Host);

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
            if self.try_alloc_infer_task(&mut infer_task, self.watermark_blocks, Device::Gpu) {
                self.try_extend_infer_task(&infer_task, 1.0, Device::Host);

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

        let stats = self.get_stats();

        {
            // Logging
            let now = utils::time::now_ns();

            if now > self.last_log_time + 5e9 as u64 {
                info!("{:}", stats.to_string());
                self.last_log_time = now;
            }
        }

        (
            infer_inputs,
            fetch_block_mappings,
            write_through_block_mappings,
            stats,
        )
    }

    pub fn update(&mut self, infer_outputs: HashMap<u64, InferOutput>) -> Vec<InferTask> {
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

                if entry.chunked {
                    continue;
                }

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

        self.allocated = still_allocated_tasks;
        self.running_batch.clear();

        finished_tasks
    }

    pub fn remove_task(&mut self, infer_task: &InferTask) {
        for seq in infer_task.get_seqs(SeqStatus::Finished) {
            self.gpu_block_manager.free(seq.seq_id);
            self.host_block_manager.free(seq.seq_id);
            self.disk_block_manager.free(seq.seq_id);
        }
    }

    pub fn clear_cache(&mut self) {
        self.gpu_block_manager.clear_cache();
        self.host_block_manager.clear_cache();
        self.disk_block_manager.clear_cache();
    }

    pub fn get_stats(&self) -> Stats {
        let num_running_reqs: usize = self.running_batch.len();
        let num_allocated_reqs: usize = self.allocated.len();
        let num_waiting_reqs: usize = self.waiting.len();
        let num_pendding_reqs: usize = self.pendding.len();
        let gpu_kv_block_usage: f32 = self.gpu_block_manager.get_block_usage();
        let host_kv_block_usage: f32 = self.host_block_manager.get_block_usage();

        Stats {
            num_running_reqs,
            num_allocated_reqs,
            num_waiting_reqs,
            num_pendding_reqs,
            gpu_kv_block_usage,
            host_kv_block_usage,
        }
    }
}
