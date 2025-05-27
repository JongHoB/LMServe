import os
import torch
import torch.distributed as dist
from llmserve_worker.models.parallel import initialize_model_parallel

from loguru import logger


def init_distributed(backend="nccl") -> None:
    """
    Initialize PyTorch distributed environment for parallelism.
    """

    assert "RANK" in os.environ, "RANK not set in environment"
    assert "WORLD_SIZE" in os.environ, "WORLD_SIZE not set in environment"
    assert "MASTER_ADDR" in os.environ, "MASTER_ADDR not set in environment"
    assert "MASTER_PORT" in os.environ, "MASTER_PORT not set in environment"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(
        os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method="env://",
    )

    logger.info(
        "[Rank {}] Initialized distributed with world_size={}, local_rank={}".
        format(rank, world_size, local_rank))

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())

    initialize_model_parallel(world_size)
    logger.info("initialize_model_parallel")
