import torch
import pycuda.driver as cuda
import numpy as np
import torch.multiprocessing as mp
import asyncio

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Any
from nixl._api import nixl_agent

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
    kv_cache_metadata: Tuple[Any, ...]
    num_layers: int
    num_gpu_blocks: int
    num_host_blocks: int
    dtype: torch.dtype
    pre_event_handles: List[bytes]
    post_event_handles: List[bytes]
    model_queue: mp.Queue
    kv_queue: mp.Queue


class KVWorker:

    def __init__(
        self,
        name: str,
        params: KVWorkerParams,
        thread_pool_size: int = 4,
    ):
        num_layers = params.num_layers
        num_gpu_blocks = params.num_gpu_blocks
        num_host_blocks = params.num_host_blocks
        torch_dtype = params.dtype

        storage = torch.UntypedStorage._new_shared_cuda(
            *params.kv_cache_metadata)
        gpu_kv_caches = torch.empty(0, dtype=torch_dtype, device='cuda')
        gpu_kv_caches.set_(storage)
        gpu_kv_caches = gpu_kv_caches.view(num_layers, num_gpu_blocks, -1)
        _, _, block_size = gpu_kv_caches.shape

        self.gpu_kv_caches = gpu_kv_caches
        self.gpu_kv_cache_ptrs = [
            self.gpu_kv_caches[layer_idx].data_ptr()
            for layer_idx in range(num_layers)
        ]

        self.ctx = cuda.Context.attach()

        np_dtype = dtype_map.get(torch_dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported dtype: {torch_dtype}")

        host_kv_caches = cuda.pagelocked_empty(
            (num_host_blocks, num_layers, block_size),
            np_dtype,
            order="C",
        )

        self.host_kv_caches = host_kv_caches
        self.host_kv_caches_tensor = torch.from_numpy(host_kv_caches)

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
        self.block_stride = block_size * self.gpu_kv_caches.element_size()

        self.dtoh_stream = cuda.Stream()
        self.htod_stream = cuda.Stream()

        kv_agent = nixl_agent(name)
        reg_desc = kv_agent.register_memory(
            self.host_kv_caches_tensor,
            is_sorted=True,
        )
        if not reg_desc:
            raise RuntimeError("KV memory registration failed")

        self.reg_desc = reg_desc
        self.kv_agent_metadata = kv_agent.get_agent_metadata()
        self.kv_agent = kv_agent
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)

    def fetch(
        self,
        block_map: BlockMapping,
        layer_idx: int,
        stream: cuda.Stream,
    ) -> None:
        gpu_kv_cache_base_ptr = self.gpu_kv_cache_ptrs[layer_idx]
        block_stride = self.block_stride

        for entry in block_map.entries:
            cuda.memcpy_htod_async(
                gpu_kv_cache_base_ptr + entry.dst_block_id * block_stride,
                self.host_kv_caches[entry.src_block_id][layer_idx],
                stream=stream,
            )

    def write_through(
        self,
        block_map: BlockMapping,
        layer_idx: int,
        stream: cuda.Stream,
    ) -> None:
        gpu_kv_cache_base_ptr = self.gpu_kv_cache_ptrs[layer_idx]
        block_stride = self.block_stride

        for entry in block_map.entries:
            cuda.memcpy_dtoh_async(
                self.host_kv_caches[entry.dst_block_id][layer_idx],
                gpu_kv_cache_base_ptr + entry.src_block_id * block_stride,
                stream=stream,
            )

    async def fetch_kv_blocks(
        self,
        fetch_block_mappings: List[BlockMapping],
        stream: cuda.Stream,
    ) -> None:
        for layer_idx in range(self.num_layers):
            if len(fetch_block_mappings) > 0:
                for block_mapping in fetch_block_mappings:
                    self.fetch(block_mapping, layer_idx, stream)
                self.kv_worker_handle.pre_events[layer_idx].record()

                self.kv_worker_handle.model_queue.put(b"")

                # This allows the fetch task to yield execution to other tasks.
                await asyncio.sleep(0)

    async def write_through_kv_blocks(
        self,
        write_through_block_mappings: List[BlockMapping],
        stream: cuda.Stream,
    ) -> None:
        for layer_idx in range(self.num_layers):
            if len(write_through_block_mappings) > 0:
                while self.kv_worker_handle.kv_queue.empty():
                    await asyncio.sleep(1e-5)
                self.kv_worker_handle.kv_queue.get()

                self.dtoh_stream.wait_for_event(
                    self.kv_worker_handle.post_events[layer_idx])
                for block_mapping in write_through_block_mappings:
                    self.write_through(block_mapping, layer_idx, stream)

    async def transfer(
        self,
        fetch_block_mappings: List[BlockMapping],
        write_through_block_mappings: List[BlockMapping],
    ) -> None:
        kv_transfer_tasks = [
            self.fetch_kv_blocks(
                fetch_block_mappings,
                self.htod_stream,
            ),
            self.write_through_kv_blocks(
                write_through_block_mappings,
                self.dtoh_stream,
            )
        ]

        await asyncio.gather(*kv_transfer_tasks)
        self.htod_stream.synchronize()
        self.dtoh_stream.synchronize()

    def get_descriptors(self, block_ids: List[int]) -> bytes:
        tensors = [
            self.host_kv_caches_tensor[block_id] for block_id in block_ids
        ]

        xfer_desc = self.kv_agent.get_xfer_descs(tensors, is_sorted=True)
        descs = self.kv_agent.get_serialized_descs(xfer_desc)

        return descs

    async def pull_kv(
        self,
        peer_name: str,
        remote_descs: bytes,
        block_ids: List[int],
    ) -> None:
        tensors = [self.host_kv_caches_tensor[bid] for bid in block_ids]

        local_descs = self.kv_agent.get_xfer_descs(tensors, is_sorted=True)
        remote_descs = self.kv_agent.deserialize_descs(remote_descs)

        xfer_handle = self.kv_agent.initialize_xfer(
            "READ",
            local_descs,
            remote_descs,
            peer_name,
            b'',
        )
        if not xfer_handle:
            print("Creating transfer failed.")
            return False

        state = self.kv_agent.transfer(xfer_handle)
        if state == "ERR":
            print("Posting transfer failed.")
            return False

        while True:
            state = self.kv_agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("Transfer Error.")
                return False
            elif state == "DONE":
                break
            else:
                await asyncio.sleep(1e-5)

        return True

    def __del__(self):
        self.ctx.pop()
        self.kv_agent.deregister_memory(self.reg_desc)
