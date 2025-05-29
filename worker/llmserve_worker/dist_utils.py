import os
import torch
import torch.distributed as dist
from llmserve_worker.models.parallel import initialize_model_parallel


def init_distributed(
    backend="nccl",
) -> None:
    """
    Initialize PyTorch distributed environment for parallelism.
    """

    assert "RANK" in os.environ, "RANK not set in environment"
    assert "WORLD_SIZE" in os.environ, "WORLD_SIZE not set in environment"
    assert "MASTER_ADDR" in os.environ, "MASTER_ADDR not set in environment"
    assert "MASTER_PORT" in os.environ, "MASTER_PORT not set in environment"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method="env://",
    )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())

    initialize_model_parallel(world_size)
