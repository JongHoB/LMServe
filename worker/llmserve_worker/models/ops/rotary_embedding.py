import triton
import triton.language as tl


@triton.jit
def rotary_embedding(query_ptr, key_ptr, position_ids_ptr, cos_cached_ptr,
                     sin_cache_ptr, rotary_dim, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    query_start = query_ptr + block_idx * rotary_dim
    key_start = key_ptr + block_idx * rotary_dim

    for i in range(0, rotary_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (rotary_dim, mask=mask)
        cos = t1.load( + offsets, mask=mask)
        sin = t1.load(
    cos =  cos_cached_ptr s


