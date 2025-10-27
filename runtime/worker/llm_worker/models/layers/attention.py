import torch
from torch import nn

import flashinfer
import triton
import triton.language as tl

from typing import Optional, Tuple

from llm_worker.models.input_params import InputParams


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


class Attention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        max_position_ids: int,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = (num_kv_heads
                             if num_kv_heads is not None else num_heads)
        self.head_size = head_size
        self.scale = float(scale)
        self.max_position_ids = max_position_ids

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_params: InputParams,
        kv_cache: Optional[Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        """
            query: [total_seq_len, hidden_size]
            key: [total_seq_len, hidden_size]
            value: [total_seq_len, hidden_size]
            kv_cache: [[num_blocks, block_len, num_heads, head_size],
                       [num_blocks, block_len, num_heads, head_size]]
        """

        # Shape: (total sequence length, num_heads, head_size)
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        output = torch.empty_like(query)
        if kv_cache is not None:
            flashinfer.append_paged_kv_cache(
                key,
                value,
                input_params.batch_indices,
                input_params.positions,
                kv_cache,
                input_params.kv_block_indices,
                input_params.kv_block_indptrs,
                input_params.kv_last_block_lens,
            )

            prefill_wrapper = input_params.prefill_wrapper
            prefill_input_len = input_params.prefill_input_len
            if prefill_wrapper is not None:
                output[:prefill_input_len] = input_params.prefill_wrapper.run(
                    query[:prefill_input_len],
                    kv_cache,
                )
            decode_wrapper = input_params.decode_wrapper
            if decode_wrapper is not None:
                output[prefill_input_len:] = input_params.decode_wrapper.run(
                    query[prefill_input_len:],
                    kv_cache,
                )
        else:
            output[:] = flashinfer.single_prefill_with_kv_cache(
                query,
                key,
                value,
                causal=True,
            )

        return output.view(-1, self.num_heads * self.head_size)
