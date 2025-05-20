import torch

from typing import Tuple

from llmserve_worker.ops.triton.rotary_embedding import apply_rotary_pos_emb_kernel


def rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = positions.shape[0]
    num_heads = query.shape[1] // head_dim
    num_kv_heads = key.shape[1] // head_dim
    rotary_dim = cos_sin_cache.shape[-1] // 2

    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)

    grid = (num_tokens, )
    apply_rotary_pos_emb_kernel[grid](
        query,
        key,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        BLOCK_SIZE=rotary_dim//2,
    )

    return q_out, k_out
