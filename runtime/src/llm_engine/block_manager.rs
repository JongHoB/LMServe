use std::cmp::min;
use std::collections::{HashMap, HashSet};

use linked_hash_set::LinkedHashSet;
use tracing::warn;

use super::hash::compute_hash;
use super::sequence::Sequence;

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

    fn clear(&mut self) {
        self.token_ids.clear();
        self.hash = Default::default();
    }
}

struct BlockAllocator {
    num_total_blocks: usize,
    block_pool: Vec<Block>,
    free_block_ids: LinkedHashSet<u32>,
}

impl BlockAllocator {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let block_pool: Vec<Block> = (0..num_blocks as u32)
            .map(|id| Block::new(id, block_size))
            .collect();
        let free_block_ids: LinkedHashSet<u32> = block_pool.iter().map(|b| b.id).collect();

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
            .pop_front()
            .expect("Cannot allocate block: all blocks are already in use.");

        let block = self
            .block_pool
            .get_mut(block_id as usize)
            .unwrap_or_else(|| panic!("Block ID {} is out of bounds", block_id));

        if block.status == BlockStatus::Used || block.ref_cnt > 0 {
            panic!("Cannot allocate block {block_id}: alread in used.");
        }

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
            assert!(
                self.free_block_ids.insert(block_id),
                "Block {block_id} is already exists in free list"
            );
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

pub type BlockHashPair = (u32, u64);

#[derive(Debug, Clone)]
pub struct BlockRegion {
    id: String,
    block_ids: Vec<u32>,
    hashes: Vec<u64>,
}

impl BlockRegion {
    fn new(block_hash_pairs: Vec<BlockHashPair>) -> Self {
        Self {
            id: utils::random::generate_id(),
            block_ids: block_hash_pairs.iter().map(|b| b.0).collect(),
            hashes: block_hash_pairs.iter().map(|b| b.1).collect(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn block_ids(&self) -> &[u32] {
        &self.block_ids
    }

    pub fn hashes(&self) -> &[u64] {
        &self.hashes
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
    reserved_blocks_table: HashMap<String, Vec<u32>>,
}

impl BlockManager {
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let block_allocator = BlockAllocator::new(block_size, num_blocks);

        Self {
            block_size,
            seq_block_mapping_table: HashMap::new(),
            hash_block_table: HashMap::new(),
            block_allocator,
            reserved_blocks_table: HashMap::new(),
        }
    }

    /// Return (block_range, last_filled_token_idx)
    pub fn get_block_ids_range(
        &self,
        seq_id: u64,
        start: usize,
        end: usize,
        include_partial_block: bool,
    ) -> Option<(Vec<u32>, usize)> {
        // Compute block range [block_start, block_end)
        let block_start = start / self.block_size;
        let mut block_end = end / self.block_size + usize::from(include_partial_block);

        if block_start >= block_end {
            return None;
        }

        let block_map = self.seq_block_mapping_table.get(&seq_id)?;

        let block_ids = &block_map.block_ids;
        if block_start >= block_ids.len() {
            return None;
        }

        block_end = block_end.min(block_ids.len());

        let last_block_id = block_ids.get(block_end - 1).unwrap();
        let last_block = self.block_allocator.get_block(*last_block_id);
        let last_token_idx = (block_end - 1) * self.block_size + last_block.token_ids.len();

        Some((
            block_ids[block_start..block_end].to_vec(),
            last_token_idx.min(block_map.filled_token_len),
        ))
    }

    pub fn get_filled_token_len(&self, seq_id: u64) -> usize {
        self.seq_block_mapping_table
            .get(&seq_id)
            .map_or(0, |block_map| block_map.filled_token_len)
    }

    pub fn get_num_required_blocks(&self, seq: &Sequence) -> usize {
        let total_token_len = seq.token_ids.len();
        let num_total_blocks = total_token_len.div_ceil(self.block_size);

        match self.seq_block_mapping_table.get(&seq.seq_id) {
            Some(_) => {
                let num_allocated_blocks = self.get_num_allocated_blocks(seq.seq_id);
                num_total_blocks - num_allocated_blocks
            }
            None => {
                let num_cached_prefix_blocks = self.get_prefix_cache_blocks(&seq.token_ids).len();
                num_total_blocks - num_cached_prefix_blocks
            }
        }
    }

    pub fn get_num_allocated_blocks(&self, seq_id: u64) -> usize {
        self.seq_block_mapping_table
            .get(&seq_id)
            .map_or(0, |block_map| block_map.block_ids.len())
    }

    pub fn get_num_allocated_unique_blocks(&self, seq_id: u64) -> usize {
        match self.seq_block_mapping_table.get(&seq_id) {
            Some(block_map) => block_map
                .block_ids
                .iter()
                .filter(|&b_id| self.block_allocator.get_block(*b_id).ref_cnt == 1)
                .count(),
            None => 0,
        }
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

    fn check_hash_collision(&self, block_id: u32, token_ids: &[u32]) -> bool {
        let block = self.block_allocator.get_block(block_id);
        if block.token_ids != token_ids {
            warn!("Hash collision occurs: token_ids mismatch detected");
            false
        } else {
            true
        }
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
                self.check_hash_collision(block_id, sub_token_ids);
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
                self.check_hash_collision(block_id, sub_token_ids);
                block_ids.push(block_id);
            } else {
                break;
            }
        }

        let matched_token_len = block_ids.len() * self.block_size;

        (block_ids, start + matched_token_len)
    }

    pub fn alloc_blocks(&mut self, seq: &Sequence) -> usize {
        let mut num_allocated_slots = self.get_num_allocated_slots(seq.seq_id);

        let token_ids = &seq.token_ids;
        let total_token_len = token_ids.len();
        let num_alloc_tokens = total_token_len - num_allocated_slots;

        let mut new_block_ids: Vec<u32> = Vec::new();

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
                    if let Some(old_block_id) = self.hash_block_table.insert(hash, last_block.id) {
                        let block_token_ids = last_block.token_ids.clone();
                        self.check_hash_collision(old_block_id, &block_token_ids);
                    }
                }

                num_allocated_slots += num_append_slots;
            }
        }

        if total_token_len > num_allocated_slots {
            for token_offset in (num_allocated_slots..total_token_len).step_by(self.block_size) {
                let token_end = min(token_offset + self.block_size, total_token_len);
                let sub_token_ids = &token_ids[token_offset..token_end];

                let block = self.block_allocator.alloc_block();
                let block_id = block.id;

                // We are going to refresh (update) the newly allocated block,
                // so we need to remove the entry of the new block that remains in the hash table
                self.hash_block_table.remove(&block.get_hash());

                block.clear();
                block.append_tokens(&mut sub_token_ids.to_vec());

                if block.get_num_empty_slots() == 0 {
                    let hash = compute_hash(&token_ids[..token_end]);
                    block.set_hash(hash);
                    if let Some(old_block_id) = self.hash_block_table.insert(hash, block_id) {
                        self.check_hash_collision(old_block_id, sub_token_ids);
                    }
                }

                new_block_ids.push(block_id);
            }
        }

        self.seq_block_mapping_table
            .entry(seq.seq_id)
            .and_modify(|block_map| {
                block_map.append_block_ids(&mut new_block_ids, num_alloc_tokens)
            })
            .or_insert(BlockMap::new(new_block_ids, num_alloc_tokens, 0));

        num_alloc_tokens
    }

    pub fn init_prefix_cache_blocks(&mut self, seq: &Sequence) -> usize {
        if self.seq_block_mapping_table.contains_key(&seq.seq_id) {
            return 0;
        }

        let mut reused_token_len = 0;
        let mut new_block_ids: Vec<u32> = Vec::new();

        for block_id in self.get_prefix_cache_blocks(&seq.token_ids) {
            self.block_allocator.alloc_block_by_id(block_id);

            reused_token_len += self.block_size;

            new_block_ids.push(block_id);
        }

        self.seq_block_mapping_table.insert(
            seq.seq_id,
            BlockMap::new(new_block_ids, reused_token_len, reused_token_len),
        );

        reused_token_len
    }

    pub fn update_prefix_cache_blocks(&mut self, seq: &Sequence) -> usize {
        let seq_id = seq.seq_id;
        let token_ids = seq.get_token_ids();

        let maybe_block_map = self.seq_block_mapping_table.get(&seq_id);
        if maybe_block_map.is_none() {
            return self.init_prefix_cache_blocks(seq);
        }

        let block_map = maybe_block_map.unwrap();

        let mut cached_token_len = block_map.filled_token_len;
        let mut new_block_ids: Vec<u32> = Vec::new();

        let (block_ids, _) =
            self.get_prefix_cache_blocks_range(token_ids, cached_token_len, token_ids.len());
        for block_id in block_ids {
            self.block_allocator.alloc_block_by_id(block_id);

            cached_token_len += self.block_size;

            new_block_ids.push(block_id);
        }

        let num_add_slots = cached_token_len - block_map.filled_token_len;

        self.seq_block_mapping_table
            .entry(seq_id)
            .and_modify(|block_map| {
                block_map.block_ids.append(&mut new_block_ids);
                block_map.num_allocated_slots = cached_token_len;
                block_map.filled_token_len = cached_token_len;
            });

        num_add_slots
    }

    pub fn reserve_blocks(&mut self, token_ids: &[u32], start: usize) -> Option<BlockRegion> {
        let start = (start / self.block_size) * self.block_size;
        let end = token_ids.len();

        let mut block_hash_pairs: Vec<BlockHashPair> = Vec::new();
        for token_offset in (start..end).step_by(self.block_size) {
            let token_end = min(token_offset + self.block_size, end);
            let sub_token_ids = &token_ids[token_offset..token_end];

            // If sub_token_ids does not fully fill a block, Do not pin it.
            if sub_token_ids.len() < self.block_size {
                break;
            }

            let block = self.block_allocator.alloc_block();
            block.clear();

            block.append_tokens(&mut sub_token_ids.to_vec());

            let hash = compute_hash(&token_ids[..token_end]);
            block.set_hash(hash);

            block_hash_pairs.push((block.id, block.get_hash()));
        }

        if block_hash_pairs.is_empty() {
            return None;
        }

        let block_region = BlockRegion::new(block_hash_pairs);
        self.reserved_blocks_table.insert(
            block_region.id().to_string(),
            block_region.block_ids().to_vec(),
        );

        Some(block_region)
    }

    pub fn activate_reserved_blocks(&mut self, region_id: &str, hash_values: &[u64]) -> usize {
        let Some(block_ids) = self.reserved_blocks_table.remove(region_id) else {
            return 0;
        };

        let block_hash_set: HashSet<u64> = hash_values.iter().copied().collect();

        let mut activated = 0usize;

        for block_id in block_ids {
            let block = self.block_allocator.get_block(block_id);
            let h = block.get_hash();

            if block_hash_set.contains(&h) {
                self.hash_block_table.entry(h).or_insert(block_id);
                activated += 1;
            }

            self.block_allocator.free_block(block_id);
        }

        activated
    }

    pub fn pin_blocks_by_hashes(&mut self, hash_values: &[u64]) -> BlockRegion {
        let mut block_hash_pairs: Vec<BlockHashPair> = Vec::new();
        for &hash in hash_values {
            if let Some(block_id) = self.lookup_block_id(hash) {
                self.block_allocator.alloc_block_by_id(block_id);
                block_hash_pairs.push((block_id, hash));
            } else {
                warn!(
                    "Block not found for hash {}. Aborting pinning buffer ({}/{})",
                    hash,
                    block_hash_pairs.len(),
                    hash_values.len(),
                );
                break;
            }
        }

        let block_region = BlockRegion::new(block_hash_pairs);
        self.reserved_blocks_table.insert(
            block_region.id().to_string(),
            block_region.block_ids().to_vec(),
        );

        block_region
    }

    pub fn unpin_blocks(&mut self, region_id: &str) -> usize {
        let Some(block_ids) = self.reserved_blocks_table.remove(region_id) else {
            return 0;
        };

        let num_blocks = block_ids.len();
        for block_id in block_ids {
            self.block_allocator.free_block(block_id);
        }

        num_blocks
    }

    pub fn update_filled_len(&mut self, seq_id: u64, new_filled_token_len: usize) {
        self.seq_block_mapping_table
            .entry(seq_id)
            .and_modify(|block_map| {
                assert!(new_filled_token_len <= block_map.num_allocated_slots);
                block_map.filled_token_len = new_filled_token_len
            });
    }

    fn drop_cache(&mut self, block_id: u32) {
        let block = self.block_allocator.get_block_mut(block_id);
        if block.ref_cnt == 0 {
            self.hash_block_table.remove(&block.get_hash());

            block.clear();
        }
    }

    /// Logically free all blocks associated with given sequence.
    /// The blocks remain available for reuse; however, if memory pressure occurs,
    /// they may be relaimed for other allocations.
    pub fn free(&mut self, seq_id: u64) {
        if let Some(block_map) = self.seq_block_mapping_table.remove(&seq_id) {
            let mut freed_token_cnt = 0;
            for block_id in block_map.block_ids.iter() {
                self.block_allocator.free_block(*block_id);

                freed_token_cnt += self.block_size;

                if freed_token_cnt > block_map.filled_token_len {
                    self.drop_cache(*block_id);
                }
            }
        }
    }

    /// Free all blocks associated with the given sequence.
    /// This function also clears cached blocks that have no remaining references,
    /// so they are no longer reused.
    pub fn free_and_drop_cache(&mut self, seq_id: u64) {
        if let Some(block_map) = self.seq_block_mapping_table.remove(&seq_id) {
            for block_id in block_map.block_ids.iter() {
                self.block_allocator.free_block(*block_id);

                self.drop_cache(*block_id);
            }
        }
    }

    pub fn get_block_usage(&self) -> f32 {
        self.block_allocator.get_block_usage()
    }

    pub fn clear_cache(&mut self) {
        self.hash_block_table.clear();
    }
}
