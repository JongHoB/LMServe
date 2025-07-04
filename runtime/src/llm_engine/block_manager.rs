use std::cmp::min;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use ahash::AHasher;
use tracing::warn;

use super::sequence::Sequence;
use utils::collections::FifoSet;

#[derive(PartialEq, Eq, Debug)]
enum BlockStatus {
    Free,
    Used,
}

#[derive(Debug)]
struct Block {
    id: u32,
    block_size: usize,
    ref_cnt: i32,
    status: BlockStatus,
    token_ids: Vec<u32>,
    hash: u64,
}

fn compute_hash(token_ids: &[u32]) -> u64 {
    let mut hasher = AHasher::default();
    token_ids.hash(&mut hasher);
    hasher.finish()
}

impl Block {
    fn new(id: u32, block_size: usize) -> Self {
        Self {
            id,
            block_size,
            ref_cnt: 0,
            status: BlockStatus::Free,
            token_ids: Vec::new(),
            hash: Default::default(),
        }
    }

    fn append_tokens(&mut self, new_token_ids: &mut Vec<u32>) {
        assert!(self.token_ids.len() + new_token_ids.len() <= self.block_size);

        self.token_ids.append(new_token_ids);
    }

    fn get_num_empty_slots(&self) -> usize {
        self.block_size - self.token_ids.len()
    }

    fn get_hash(&self) -> u64 {
        self.hash
    }

    fn set_hash(&mut self, hash: u64) {
        self.hash = hash;
    }
}

struct BlockAllocator {
    num_total_blocks: usize,
    block_pool: Vec<Block>,
    free_block_ids: FifoSet<u32>,
}

// Make BlockAllocator methods return `Option<T>` instead of panicking.
impl BlockAllocator {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let block_pool: Vec<Block> = (0..num_blocks as u32)
            .map(|id| Block::new(id, block_size))
            .collect();
        let free_block_ids: FifoSet<u32> = block_pool.iter().map(|b| b.id).collect();

        Self {
            num_total_blocks: num_blocks,
            block_pool,
            free_block_ids,
        }
    }

    fn get_block(&self, block_id: u32) -> &Block {
        self.block_pool
            .get(block_id as usize)
            .unwrap_or_else(|| panic!("Block ID {} is out of bounds", block_id))
    }

    fn get_block_mut(&mut self, block_id: u32) -> &mut Block {
        self.block_pool
            .get_mut(block_id as usize)
            .unwrap_or_else(|| panic!("Block ID {} is out of bounds", block_id))
    }

    fn can_alloc_blocks(&self, num_required_blocks: usize, watermark: f32) -> bool {
        assert!(
            0.0 < watermark && watermark <= 1.0,
            "watermark must be in (0, 1]"
        );
        let num_free_blocks = self.free_block_ids.len();
        let num_used_blocks = self.num_total_blocks - num_free_blocks;

        num_used_blocks + num_required_blocks <= (watermark * self.num_total_blocks as f32) as usize
    }

    fn alloc_block(&mut self) -> &mut Block {
        let block_id = self
            .free_block_ids
            .pop()
            .expect("Cannot allocate block: all blocks are already in use.");

        let block = self
            .block_pool
            .get_mut(block_id as usize)
            .unwrap_or_else(|| panic!("Block ID {} is out of bounds", block_id));

        if block.status == BlockStatus::Used {
            panic!("Cannot allocate block {block_id}: alread in used.");
        }

        block.token_ids.clear();
        block.hash = Default::default();

        block.status = BlockStatus::Used;
        block.ref_cnt = 1;
        block
    }

    fn free_block(&mut self, block_id: u32) {
        let block = self.get_block_mut(block_id);
        assert!(
            block.status != BlockStatus::Free || block.ref_cnt <= 0,
            "Double free detected for block {}.",
            block_id
        );
        block.ref_cnt -= 1;
        if block.ref_cnt == 0 {
            block.status = BlockStatus::Free;
            self.free_block_ids.insert(block_id);
        }
    }

    fn alloc_block_by_id(&mut self, block_id: u32) -> &mut Block {
        let block = self
            .block_pool
            .get_mut(block_id as usize)
            .unwrap_or_else(|| panic!("Block ID {} is out of bounds", block_id));

        if block.status == BlockStatus::Free {
            assert!(
                self.free_block_ids.remove(&block_id),
                "BrokenAllocatorError: block {block_id} is free but not in free list"
            );
            block.status = BlockStatus::Used;
        }
        block.ref_cnt += 1;
        block
    }

    fn get_block_usage(&self) -> f32 {
        let num_free_blocks = self.free_block_ids.len();
        let num_use_blocks = self.num_total_blocks - num_free_blocks;

        num_use_blocks as f32 / self.num_total_blocks as f32
    }
}

struct BlockMap {
    block_ids: Vec<u32>,
    num_allocated_slots: usize,
    filled_token_len: usize,
}

impl BlockMap {
    fn new(block_ids: Vec<u32>, num_allocated_slots: usize, filled_token_len: usize) -> Self {
        Self {
            block_ids,
            num_allocated_slots,
            filled_token_len,
        }
    }

    fn append_block_ids(&mut self, block_ids: &mut Vec<u32>, num_allocated_slots: usize) {
        self.block_ids.append(block_ids);
        self.num_allocated_slots += num_allocated_slots;
    }
}

pub struct BlockManager {
    pub block_size: usize,
    seq_block_mapping_table: HashMap<u64, BlockMap>,
    hash_block_table: HashMap<u64, u32>, // HashMap<hash value, block id>
    block_allocator: BlockAllocator,
    seq_block_buffer: HashMap<u64, Vec<u32>>,
}

impl BlockManager {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let block_allocator = BlockAllocator::new(block_size, num_blocks);

        Self {
            block_size,
            seq_block_mapping_table: HashMap::new(),
            hash_block_table: HashMap::new(),
            block_allocator,
            seq_block_buffer: HashMap::new(),
        }
    }

    pub fn get_block_ids_range(&self, seq_id: u64, start: usize, end: usize) -> (Vec<u32>, usize) {
        let block_offset = start / self.block_size;
        let mut block_end = end / self.block_size + 1;

        if let Some(block_map) = self.seq_block_mapping_table.get(&seq_id) {
            let block_ids = &block_map.block_ids;
            if block_offset >= block_end || block_offset >= block_ids.len() {
                return (Vec::new(), block_map.filled_token_len);
            }

            block_end = block_end.min(block_ids.len());

            (
                block_ids[block_offset..block_end].to_vec(),
                block_map.filled_token_len,
            )
        } else {
            (Vec::new(), 0)
        }
    }

    pub fn get_filled_token_len(&self, seq_id: u64) -> usize {
        self.seq_block_mapping_table
            .get(&seq_id)
            .map_or(0, |block_map| block_map.filled_token_len)
    }

    pub fn get_num_required_blocks(&self, seq: &Sequence) -> usize {
        let filled_token_len = self.get_filled_token_len(seq.seq_id);
        let total_token_len = seq.token_ids.len();
        let num_required_slots = total_token_len.saturating_sub(filled_token_len);
        num_required_slots.div_ceil(self.block_size)
    }

    fn get_last_block_id(&self, seq_id: u64) -> Option<u32> {
        let block_map = self.seq_block_mapping_table.get(&seq_id)?;
        let last_block_id = block_map.block_ids.last()?;
        Some(*last_block_id)
    }

    pub fn get_num_allocated_slots(&self, seq_id: u64) -> usize {
        self.seq_block_mapping_table
            .get(&seq_id)
            .map_or(0, |block_map| block_map.num_allocated_slots)
    }

    pub fn can_alloc_seq(&self, seq: &Sequence, watermark: f32) -> bool {
        let num_blocks = self.get_num_required_blocks(seq);
        self.block_allocator.can_alloc_blocks(num_blocks, watermark)
    }

    pub fn can_alloc_blocks(&self, num_blocks: usize, watermark: f32) -> bool {
        self.block_allocator.can_alloc_blocks(num_blocks, watermark)
    }

    fn lookup_block_id(&self, hash: u64) -> Option<u32> {
        if let Some(&block_id) = self.hash_block_table.get(&hash) {
            let block = self.block_allocator.get_block(block_id);
            if block.get_hash() == hash {
                return Some(block_id);
            } else {
                return None;
            }
        }

        None
    }

    pub fn get_prefix_cache_blocks(&self, token_ids: &[u32]) -> Vec<u32> {
        let total_token_len = token_ids.len();

        let mut block_ids: Vec<u32> = Vec::new();
        for token_offset in (0..total_token_len).step_by(self.block_size) {
            let token_end = min(token_offset + self.block_size, total_token_len);
            let sub_token_ids = &token_ids[token_offset..token_end];

            if sub_token_ids.len() < self.block_size {
                break;
            }

            let hash = compute_hash(&token_ids[..token_end]);

            if let Some(block_id) = self.lookup_block_id(hash) {
                let block = self.block_allocator.get_block(block_id);
                if block.token_ids != sub_token_ids {
                    warn!("Hash collision on lookup: token_ids mismatch detected");
                }
                block_ids.push(block_id);
            } else {
                break;
            }
        }

        block_ids
    }

    pub fn get_prefix_cache_blocks_range(
        &self,
        token_ids: &[u32],
        start: usize,
        end: usize,
    ) -> (Vec<u32>, usize) {
        let start = start.div_ceil(self.block_size) * self.block_size;
        let end = end.min(token_ids.len());
        assert!(start <= end);

        let mut block_ids: Vec<u32> = Vec::new();
        for token_offset in (start..end).step_by(self.block_size) {
            let token_end = min(token_offset + self.block_size, end);
            let sub_token_ids = &token_ids[token_offset..token_end];

            if sub_token_ids.len() < self.block_size {
                break;
            }

            let hash = compute_hash(&token_ids[..token_end]);

            if let Some(block_id) = self.lookup_block_id(hash) {
                let block = self.block_allocator.get_block(block_id);
                if block.token_ids != sub_token_ids {
                    warn!("Hash collision on lookup: token_ids mismatch detected");
                }
                block_ids.push(block_id);
            } else {
                break;
            }
        }

        let matched_token_len = block_ids.len() * self.block_size;

        (block_ids, start + matched_token_len)
    }

    fn alloc_blocks(&mut self, seq: &Sequence) -> usize {
        let mut num_allocated_slots = self.get_num_allocated_slots(seq.seq_id);

        let token_ids = &seq.token_ids;
        let total_token_len = token_ids.len();
        let num_alloc_tokens = total_token_len - num_allocated_slots;

        let mut add_block_ids: Vec<u32> = Vec::new();

        if num_allocated_slots > 0 {
            let last_block_id = self
                .get_last_block_id(seq.seq_id)
                .expect("Not found a last block");
            let last_block = self.block_allocator.get_block_mut(last_block_id);

            let num_append_slots = min(last_block.get_num_empty_slots(), num_alloc_tokens);
            if num_append_slots > 0 {
                let token_end = num_allocated_slots + num_append_slots;
                let sub_token_ids = &mut (token_ids[num_allocated_slots..token_end]).to_vec();
                last_block.append_tokens(sub_token_ids);

                if last_block.get_num_empty_slots() == 0 {
                    let hash = compute_hash(&token_ids[..token_end]);
                    last_block.set_hash(hash);
                    self.hash_block_table.insert(hash, last_block.id);
                }

                num_allocated_slots += num_append_slots;
            }
        }

        if total_token_len > num_allocated_slots {
            for token_offset in (num_allocated_slots..total_token_len).step_by(self.block_size) {
                let token_end = min(token_offset + self.block_size, total_token_len);
                let sub_token_ids = &token_ids[token_offset..token_end];

                let block = self.block_allocator.alloc_block();

                // We are going to refresh (update) the newly allocated block,
                // so we need to remove the entry of the new block that remains in the hash table
                self.hash_block_table.remove(&block.get_hash());

                block.append_tokens(&mut sub_token_ids.to_vec());

                if block.get_num_empty_slots() == 0 {
                    let hash = compute_hash(&token_ids[..token_end]);
                    block.set_hash(hash);
                    self.hash_block_table.insert(hash, block.id);
                }

                add_block_ids.push(block.id);
            }
        }

        self.seq_block_mapping_table
            .entry(seq.seq_id)
            .and_modify(|block_map| {
                block_map.append_block_ids(&mut add_block_ids, num_alloc_tokens)
            })
            .or_insert(BlockMap::new(add_block_ids, num_alloc_tokens, 0));

        num_alloc_tokens
    }

    pub fn init_prefix_cache_blocks(&mut self, seq: &Sequence) -> usize {
        if self.seq_block_mapping_table.contains_key(&seq.seq_id) {
            return 0;
        }

        let mut reused_token_len = 0;
        let mut add_block_ids: Vec<u32> = Vec::new();

        for block_id in self.get_prefix_cache_blocks(&seq.token_ids) {
            self.block_allocator.alloc_block_by_id(block_id);

            reused_token_len += self.block_size;

            add_block_ids.push(block_id);
        }

        self.seq_block_mapping_table.insert(
            seq.seq_id,
            BlockMap::new(add_block_ids, reused_token_len, reused_token_len),
        );

        reused_token_len
    }

    pub fn hold_seq_tokens(
        &mut self,
        seq_id: u64,
        token_ids: &[u32],
        start: usize,
        end: usize,
    ) -> Vec<u32> {
        let start = start.div_ceil(self.block_size) * self.block_size;
        let end = end.min(token_ids.len());
        assert!(start <= end);

        let mut new_block_ids: Vec<u32> = Vec::new();

        for token_offset in (start..end).step_by(self.block_size) {
            let token_end = min(token_offset + self.block_size, end);
            let sub_token_ids = &token_ids[token_offset..token_end];

            let block = self.block_allocator.alloc_block();

            block.append_tokens(&mut sub_token_ids.to_vec());

            if block.get_num_empty_slots() == 0 {
                let hash = compute_hash(&token_ids[..token_end]);
                block.set_hash(hash);
            }

            new_block_ids.push(block.id);
        }

        self.seq_block_buffer.insert(seq_id, new_block_ids.clone());

        new_block_ids
    }

    pub fn release_seq_tokens(&mut self, seq_id: u64) {
        if let Some(block_ids) = self.seq_block_buffer.remove(&seq_id) {
            for block_id in block_ids {
                let block = self.block_allocator.get_block(block_id);
                self.hash_block_table.insert(block.get_hash(), block_id);
                self.block_allocator.free_block(block_id);
            }
        }
    }

    pub fn get_num_prefix_cache_blocks(&self, token_ids: &[u32]) -> usize {
        self.get_prefix_cache_blocks(token_ids).len()
    }

    pub fn get_prefix_cache_token_len(&self, token_ids: &[u32]) -> usize {
        self.get_prefix_cache_blocks(token_ids).len() * self.block_size
    }

    pub fn reserve_blocks(&mut self, seq: &Sequence) -> usize {
        self.alloc_blocks(seq)
    }

    pub fn update_filled_len(&mut self, seq_id: u64, new_filled_token_len: usize) {
        self.seq_block_mapping_table
            .entry(seq_id)
            .and_modify(|block_map| {
                assert!(new_filled_token_len <= block_map.num_allocated_slots);
                block_map.filled_token_len = new_filled_token_len
            });
    }

    pub fn free(&mut self, seq_id: u64) {
        if let Some(block_map) = self.seq_block_mapping_table.remove(&seq_id) {
            for block_id in block_map.block_ids.iter() {
                self.block_allocator.free_block(*block_id);
            }
        }
    }

    pub fn get_block_usage(&self) -> f32 {
        self.block_allocator.get_block_usage()
    }
}
