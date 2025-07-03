import triton
import triton.language as tl


@triton.jit
def copy_blocks_kernel(
    kv_cache_ptr,
    block_maps_ptr,
    numel_per_layer,
    numel_per_block,
    BLOCK_SIZE: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    numel_per_layer = tl.cast(numel_per_layer, dtype=tl.int64)
    numel_per_block = tl.cast(numel_per_block, dtype=tl.int64)

    src_block_id = tl.cast(
        tl.load(block_maps_ptr + (block_idx * 2)),
        dtype=tl.int64,
    )
    dst_block_id = tl.cast(
        tl.load(block_maps_ptr + (block_idx * 2 + 1)),
        dtype=tl.int64,
    )

    layer_kv_ptr = kv_cache_ptr + (layer_idx * numel_per_layer)
    src_offset = src_block_id * numel_per_block
    dst_offset = dst_block_id * numel_per_block

    for i in range(0, numel_per_block, BLOCK_SIZE):
        idxs = i + tl.arange(0, BLOCK_SIZE)
        mask = idxs < (numel_per_block)
        data = tl.load(layer_kv_ptr + src_offset + idxs, mask=mask)
        tl.store(layer_kv_ptr + dst_offset + idxs, data, mask=mask)


@triton.jit
def fill_key_cache_kernel(key_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                          slot_mapping_ptr, num_kv_heads, head_size,
                          BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)

    key_start = key_ptr + block_idx * num_kv_heads * head_size
    value_start = value_ptr + block_idx * num_kv_heads * head_size

    slot_idx = tl.cast(tl.load(slot_mapping_ptr + block_idx), dtype=tl.int64)
    key_cache_start = key_cache_ptr + slot_idx * num_kv_heads * head_size
    value_cache_start = value_cache_ptr + slot_idx * num_kv_heads * head_size

    for i in range(0, num_kv_heads * head_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (num_kv_heads * head_size)
        key = tl.load(key_start + offsets, mask=mask)
        value = tl.load(value_start + offsets, mask=mask)
        tl.store(key_cache_start + offsets, key, mask=mask)
        tl.store(value_cache_start + offsets, value, mask=mask)
