import torch
import pycuda.driver as cuda
import numpy as np
import torch.multiprocessing as mp

from dataclasses import dataclass
from typing import List

from llmserve_worker.pb.worker_pb2 import BlockMapping

dtype_map = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.uint16,
    torch.int32: np.int32,
}


@dataclass
class KVWorkerHandle:
    pre_events: List[cuda.Event]
    post_events: List[cuda.Event]
    model_queue: mp.Queue
    kv_queue: mp.Queue


@dataclass
class KVWorkerParams:
    kv_cache_metadata: torch.Tensor
    num_layers: int
    num_blocks: int
    dtype: torch.dtype
    pre_event_handles: List[bytes]
    post_event_handles: List[bytes]
    model_queue: mp.Queue
    kv_queue: mp.Queue


class KVWorker:

    def __init__(self, params: KVWorkerParams):
        storage = torch.UntypedStorage._new_shared_cuda(
            *params.kv_cache_metadata)
        kv_caches = torch.empty(0, dtype=params.dtype, device='cuda')
        kv_caches.set_(storage)
        kv_caches = kv_caches.view(params.num_layers, params.num_blocks, -1)

        self.kv_caches = kv_caches
        torch_dtype = kv_caches.dtype
        np_dtype = dtype_map.get(torch_dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype: {torch_dtype}")

        self.ctx = cuda.Context.attach()

        # This block_size does not refer to PagedAttention's block size,
        # but to the total number of elements in a block
        # (block_size × hidden_dim).
        block_size = kv_caches.stride(1)

        self.host_kv_caches = cuda.pagelocked_empty(
            (params.num_blocks, params.num_layers, block_size),
            np_dtype,
            order="C",
        )

        pre_events = [
            cuda.Event.from_ipc_handle(handle)
            for handle in params.pre_event_handles
        ]
        post_events = [
            cuda.Event.from_ipc_handle(handle)
            for handle in params.post_event_handles
        ]

        self.kv_worker_handle = KVWorkerHandle(
            pre_events,
            post_events,
            params.model_queue,
            params.kv_queue,
        )

        self.num_layers = params.num_layers
        self.block_stride = block_size * kv_caches.element_size()

        self.stream = cuda.Stream()

    def transfer(self, block_mappings: List[BlockMapping]) -> None:
        for layer_idx in range(self.num_layers):
            dst = self.kv_caches[layer_idx].data_ptr()

            self.kv_worker_handle.kv_queue.get()
            self.stream.wait_for_event(
                self.kv_worker_handle.post_events[layer_idx])
            for block_mapping in block_mappings:
                for block_id in block_mapping.block_ids:
                    cuda.memcpy_dtoh_async(
                        self.host_kv_caches[block_id, layer_idx],
                        dst + block_id * self.block_stride,
                        stream=self.stream,
                    )

        self.stream.synchronize()

    def __del__(self):
        self.ctx.pop()
