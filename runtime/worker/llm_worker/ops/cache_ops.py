import torch

from typing import Dict, List

from llm_worker.ops.triton import cache_kernels


def copy_blocks(
    kv_caches: torch.Tensor,
    src_to_dst: Dict[int, List[int]],
) -> None:
    block_maps = []
    for src, dsts in src_to_dst.items():
        for dst in dsts:
            block_maps += [src, dst]

    block_maps_tensor = torch.tensor(block_maps,
                                     dtype=torch.long,
                                     device='cuda')

    num_layers = kv_caches.shape[0]
    numel_per_layer = kv_caches[0].numel()
    numel_per_block = kv_caches[0][0].numel()

    num_copy_blocks = len(block_maps) // 2
    grid = (num_layers, num_copy_blocks, )
    cache_kernels.copy_blocks_kernel[grid](
        kv_caches,
        block_maps_ptr=block_maps_tensor,
        numel_per_layer=numel_per_layer,
        numel_per_block=numel_per_block,
        BLOCK_SIZE=1024,
    )


def fill_key_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_kv_heads = key.shape[1]
    head_size = key.shape[2]

    grid = (len(slot_mapping), )
    cache_kernels.fill_key_cache_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        num_kv_heads,
        head_size,
        BLOCK_SIZE=1024,
    )
