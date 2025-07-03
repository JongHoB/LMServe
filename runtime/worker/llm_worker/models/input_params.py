import torch

from typing import Optional
from dataclasses import dataclass
from flashinfer import (BatchPrefillWithPagedKVCacheWrapper,
                        BatchDecodeWithPagedKVCacheWrapper)


@dataclass
class InputParams:
    position_ids: torch.Tensor
    cu_seqlens_q: Optional[torch.Tensor] = None
    prefill_input_len: Optional[int] = None

    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None

    batch_indices: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    kv_block_indices: Optional[torch.Tensor] = None
    kv_block_indptrs: Optional[torch.Tensor] = None
    kv_last_block_lens: Optional[torch.Tensor] = None
