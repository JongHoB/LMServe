import triton
import triton.language as tl


@triton.jit
def apply_rotary_pos_emb_kernel(
    query_ptr,
    key_ptr,
    position_ptr,
    cos_sin_cache_ptr,  # [max_position, 2 * rotary_dim]
    out_query_ptr,
    out_key_ptr,
    num_tokens,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    pos = tl.load(position_ptr + token_idx)

    cos_ptr = cos_sin_cache_ptr + pos * (rotary_dim * 2)
    sin_ptr = cos_sin_cache_ptr + pos * (rotary_dim * 2) + rotary_dim

    query_ptr += token_idx * (num_heads * head_dim)
    key_ptr += token_idx * (num_kv_heads * head_dim)
    out_query_ptr += token_idx * (num_heads * head_dim)
    out_key_ptr += token_idx * (num_kv_heads * head_dim)

    embed_dim = rotary_dim // 2
    for head_id in range(0, num_heads):
        head_idx = head_id * head_dim

        for i in range(0, embed_dim, BLOCK_SIZE):
            idxs = i + tl.arange(0, BLOCK_SIZE)
            mask = (idxs < embed_dim)

            x_idx = idxs
            y_idx = embed_dim + idxs

            cos_x = tl.load(cos_ptr + x_idx, mask=mask)
            cos_y = tl.load(cos_ptr + y_idx, mask=mask)

            sin_x = tl.load(sin_ptr + x_idx, mask=mask)
            sin_y = tl.load(sin_ptr + y_idx, mask=mask)

            q_x = tl.load(query_ptr + head_idx + x_idx, mask=mask)
            q_y = tl.load(query_ptr + head_idx + y_idx, mask=mask)

            out_query_x = q_x * cos_x - q_y * sin_x
            out_query_y = q_y * cos_y + q_x * sin_y

            tl.store(out_query_ptr + head_idx + x_idx, out_query_x, mask=mask)
            tl.store(out_query_ptr + head_idx + y_idx, out_query_y, mask=mask)

            if head_id < num_kv_heads:
                k_x = tl.load(key_ptr + head_idx + x_idx, mask=mask)
                k_y = tl.load(key_ptr + head_idx + y_idx, mask=mask)

                out_key_x = k_x * cos_x - k_y * sin_x
                out_key_y = k_y * cos_y + k_x * sin_y

                tl.store(out_key_ptr + head_idx + x_idx, out_key_x, mask=mask)
                tl.store(out_key_ptr + head_idx + y_idx, out_key_y, mask=mask)
