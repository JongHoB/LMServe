import torch
import pycuda.driver as cuda
import numpy as np
import torch.multiprocessing as mp
import asyncio
import time
import os

from loguru import logger
from dataclasses import dataclass
from typing import List, Tuple, Any
from nixl._api import nixl_agent, nixl_agent_config
from concurrent.futures import ThreadPoolExecutor

from llm_worker.pb.worker_pb2 import BlockMapping, Device

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
    num_disk_blocks: int
    disk_kv_cache_file: str
    dtype: torch.dtype
    pre_event_handles: List[bytes]
    post_event_handles: List[bytes]
    model_queue: mp.Queue
    kv_queue: mp.Queue


def memmap_disk_kv_cache(
    kv_cache_file: str,
    shape: Tuple,
    dtype: np.dtype,
    mode: str = "r+",
):
    need_new = False
    if os.path.exists(kv_cache_file):
        try:
            kv_cache = np.load(kv_cache_file, mmap_mode=mode)
            if kv_cache.shape != shape or kv_cache.dtype != np.dtype(dtype):
                logger.warning(
                    "Disk KV cache config changed "
                    "(expected shape={} dtype={}, got shape={} dtype={})".
                    format(kv_cache.shape, kv_cache.dtype, shape,
                           np.dtype(dtype)))
                kv_cache._mmap.close()
                os.remove(kv_cache_file)
                need_new = True
            else:
                return kv_cache

        except Exception as e:
            logger.warning(f"Failed to load disk KV cache file: {e}")
            os.remove(kv_cache_file)
            need_new = True

    else:
        need_new = True

    if need_new:
        logger.info(f"Creating new disk kv caches file: {kv_cache_file}")
        kv_cache = np.lib.format.open_memmap(kv_cache_file,
                                             mode="w+",
                                             dtype=dtype,
                                             shape=shape)

        return kv_cache


class KVWorker:

    def __init__(
        self,
        name: str,
        params: KVWorkerParams,
        num_disk_copy_workers: int = 4,
        num_nixl_transfer_workers: int = 4,
    ):
        num_layers = params.num_layers
        num_gpu_blocks = params.num_gpu_blocks
        num_host_blocks = params.num_host_blocks
        num_disk_blocks = params.num_disk_blocks
        disk_kv_cache_file = params.disk_kv_cache_file
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

        if num_disk_blocks > 0:
            disk_kv_caches_shape = (num_disk_blocks, num_layers, block_size)
            self.disk_kv_caches = memmap_disk_kv_cache(disk_kv_cache_file,
                                                       disk_kv_caches_shape,
                                                       np_dtype)
        else:
            self.disk_kv_caches = None

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

        # Initialize NIXL agent with a dummy POSIX backend.
        # This backend is not actively used in the current setup.
        nixl_config = nixl_agent_config(backends=["POSIX"])
        kv_agent = nixl_agent(name, nixl_config)

        # Create the actual UCX backend with the specified number of workers.
        kv_agent.create_backend("UCX", {
            "num_workers": str(num_nixl_transfer_workers),
        })

        reg_desc = kv_agent.register_memory(
            self.host_kv_caches_tensor,
            backends=["UCX"],
        )
        if not reg_desc:
            raise RuntimeError("KV memory registration failed")

        self.reg_desc = reg_desc
        self.kv_agent_name = name
        self.kv_agent_metadata = kv_agent.get_agent_metadata()
        self.kv_agent = kv_agent

        self.nixl_thread_pool = ThreadPoolExecutor(
            max_workers=num_nixl_transfer_workers)

        self.disk_thread_pool = ThreadPoolExecutor(
            max_workers=num_disk_copy_workers)

        self.COPY_KV_OP_MAP = {
            (Device.GPU, Device.Host): self._copy_kv_gpu_to_host,
            (Device.Host, Device.GPU): self._copy_kv_host_to_gpu,
            (Device.Host, Device.Disk): self._copy_kv_host_to_disk,
            (Device.Disk, Device.Host): self._copy_kv_disk_to_host,
        }

    def _load_layer_kv_blocks_from_host(
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

    def _save_layer_kv_blocks_to_host(
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

    def _load_layer_kv_blocks_from_disk(
        self,
        block_mappings: List[BlockMapping],
        layer_idx: int,
    ) -> None:
        for block_map in block_mappings:
            for entry in block_map.entries:
                self.host_kv_caches[entry.dst_block_id][layer_idx] = (
                    self.disk_kv_caches[entry.src_block_id][layer_idx])

    def _save_layer_kv_blocks_to_disk(
        self,
        block_mappings: List[BlockMapping],
        layer_idx: int,
    ) -> None:
        for block_map in block_mappings:
            for entry in block_map.entries:
                self.disk_kv_caches[entry.dst_block_id][layer_idx] = (
                    self.host_kv_caches[entry.src_block_id][layer_idx])

    async def load_kv_blocks_from_host(
        self,
        block_mappings: List[BlockMapping],
        stream: cuda.Stream,
    ) -> None:
        for layer_idx in range(self.num_layers):
            if len(block_mappings) > 0:
                for block_mapping in block_mappings:
                    self._load_layer_kv_blocks_from_host(
                        block_mapping,
                        layer_idx,
                        stream,
                    )

                    # This allows the task to yield execution to other tasks.
                    await asyncio.sleep(0)

                self.kv_worker_handle.pre_events[layer_idx].record(stream)

                self.kv_worker_handle.model_queue.put(b"")

        await asyncio.to_thread(stream.synchronize)

    async def save_kv_blocks_to_host(
        self,
        block_mappings: List[BlockMapping],
        stream: cuda.Stream,
    ) -> None:
        for layer_idx in range(self.num_layers):
            if len(block_mappings) > 0:
                while self.kv_worker_handle.kv_queue.empty():
                    await asyncio.sleep(1e-5)
                self.kv_worker_handle.kv_queue.get()

                stream.wait_for_event(
                    self.kv_worker_handle.post_events[layer_idx])
                for block_mapping in block_mappings:
                    self._save_layer_kv_blocks_to_host(
                        block_mapping,
                        layer_idx,
                        stream,
                    )

                    # This allows the task to yield execution to other tasks.
                    await asyncio.sleep(0)

        await asyncio.to_thread(stream.synchronize)

    async def load_kv_blocks_from_disk(
        self,
        block_mappings: List[BlockMapping],
    ):
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                self.disk_thread_pool,
                self._load_layer_kv_blocks_from_disk,
                block_mappings,
                i,
            ) for i in range(self.num_layers)
        ]
        await asyncio.gather(*tasks)

    async def save_kv_blocks_to_disk(self, block_mappings: List[BlockMapping]):
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                self.disk_thread_pool,
                self._save_layer_kv_blocks_to_disk,
                block_mappings,
                i,
            ) for i in range(self.num_layers)
        ]
        await asyncio.gather(*tasks)

    async def _copy_kv_gpu_to_host(self, block_mappings: List[BlockMapping]):
        await self.save_kv_blocks_to_host(block_mappings, self.dtoh_stream)

    async def _copy_kv_host_to_gpu(self, block_mappings: List[BlockMapping]):
        await self.load_kv_blocks_from_host(block_mappings, self.htod_stream)

    async def _copy_kv_host_to_disk(self, block_mappings: List[BlockMapping]):
        await self.save_kv_blocks_to_disk(block_mappings)

    async def _copy_kv_disk_to_host(self, block_mappings: List[BlockMapping]):
        await self.load_kv_blocks_from_disk(block_mappings)

    async def copy_kv(
        self,
        block_mappings: List[BlockMapping],
        src: Device,
        dst: Device,
    ) -> None:
        key = (src, dst)
        if key not in self.COPY_KV_OP_MAP:
            raise ValueError(f"Unsupported copy direction: {src} -> {dst}")

        await self.COPY_KV_OP_MAP[(src, dst)](block_mappings)

    def get_descriptors(self, block_ids: List[int]) -> bytes:
        tensors = [
            self.host_kv_caches_tensor[block_id] for block_id in block_ids
        ]

        xfer_desc = self.kv_agent.get_xfer_descs(tensors)
        descs = self.kv_agent.get_serialized_descs(xfer_desc)

        return descs

    async def push_kv(
        self,
        peer_name: str,
        remote_descs: bytes,
        block_ids: List[int],
    ) -> bool:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.nixl_thread_pool,
            self._push_kv,
            peer_name,
            remote_descs,
            block_ids,
        )

        return result

    def _push_kv(
        self,
        peer_name: str,
        remote_descs: bytes,
        block_ids: List[int],
    ) -> bool:
        tensors = [self.host_kv_caches_tensor[bid] for bid in block_ids]

        local_descs = self.kv_agent.get_xfer_descs(tensors)
        remote_descs = self.kv_agent.deserialize_descs(remote_descs)

        while remote_descs.descCount() > len(block_ids):
            remote_descs.remDesc(remote_descs.descCount() - 1)

        xfer_handle = self.kv_agent.initialize_xfer(
            "WRITE",
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
                time.sleep(1e-5)

        return True

    async def pull_kv(
        self,
        peer_name: str,
        remote_descs: bytes,
        block_ids: List[int],
    ) -> None:
        tensors = [self.host_kv_caches_tensor[bid] for bid in block_ids]

        local_descs = self.kv_agent.get_xfer_descs(tensors)
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
        self.kv_agent.deregister_memory(self.reg_desc)

        # Release CUDA shared memory
        del self.gpu_kv_caches
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        self.ctx.pop()
