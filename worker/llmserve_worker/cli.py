import os
import typer
import grpc
import asyncio
import signal
import torch

from loguru import logger

from llmserve_worker import ModelWorker
from llmserve_worker.dist_utils import init_distributed
from llmserve_worker.pb import worker_pb2_grpc
from llmserve_worker.pb.worker_pb2 import (
    WarmupRequest,
    WarmupResponse,
    InferRequest,
    InferResponse,
    InitCacheRequest,
    InitCacheResponse,
)

app = typer.Typer()


class WorkerService(worker_pb2_grpc.WorkerServicer):

    def __init__(self, worker: ModelWorker):
        self.worker = worker

    async def Warmup(
        self,
        request: WarmupRequest,
        context: grpc.aio.ServicerContext,
    ) -> WarmupResponse:
        max_batch_size = request.max_batch_size
        max_seq_len = request.max_seq_len
        max_num_batched_tokens = request.max_num_batched_tokens

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.worker.warmup(
            max_batch_size,
            max_seq_len,
            max_num_batched_tokens,
        )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        _, total = torch.cuda.mem_get_info()

        return WarmupResponse(gpu_total_mem_size=total, gpu_peak_mem_size=peak)

    async def Infer(
        self,
        request: InferRequest,
        context: grpc.aio.ServicerContext,
    ) -> InferResponse:
        inputs = request.inputs
        use_cache = request.use_cache

        outputs = self.worker.execute(inputs, use_cache)

        return InferResponse(outputs=outputs)

    async def InitCache(
        self,
        request: InitCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> InitCacheResponse:
        cache_size = request.cache_size

        num_blocks = self.worker.init_cache(cache_size)

        return InitCacheResponse(num_blocks=num_blocks)


@app.command()
def serve(
    model_name: str,
    block_size: int,
    device: int,
    uds_path: str,
):
    torch.cuda.set_device(device)

    init_distributed()

    logger.info(f"Launching {model_name} model on GPU {device}...")
    worker = ModelWorker(model_name, block_size)

    async def serve_inner(worker: ModelWorker, uds_path: str):

        server = grpc.aio.server()
        worker_pb2_grpc.add_WorkerServicer_to_server(WorkerService(worker),
                                                     server)
        address = f"unix://{uds_path}"
        server.add_insecure_port(address)

        logger.info(f"Model worker is ready and listening on: {address}")

        await server.start()

        shutdown_event = asyncio.Event()

        def handle_signal():
            logger.info("Shutdown signal received.")
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        await shutdown_event.wait()

        logger.info("Shutting the model worker.")

    asyncio.run(serve_inner(worker, uds_path))


if __name__ == "__main__":
    app()
