import torch
import torch.distributed as dist
import subprocess
import os

from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from typing import Optional, List

from llm_worker.models.parallel import (initialize_model_parallel,
                                        destroy_model_parallel)


@dataclass
class PCIBus:
    gpu_idx: int
    domain: str
    number: str

    def path(self) -> Path:
        domain = self.domain.removeprefix('0x').lower()
        number = self.number.removeprefix('0x').lower()
        return Path(f"/sys/class/pci_bus/{domain}:{number}/")

    def cpu_affinity(self) -> Optional[str]:
        affinity_file = self.path() / "cpulistaffinity"
        try:
            return affinity_file.read_text().strip()
        except FileNotFoundError:
            logger.warning("Not found CPU affinity file for"
                           f"GPU {self.gpu_idx} (path: {affinity_file})")
            return None

    def cpu_affinity_ids(self) -> Optional[List[int]]:
        affinity_str = self.cpu_affinity()
        if affinity_str is None:
            return None

        cpu_ids = []
        for part in affinity_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                cpu_ids.extend(range(start, end + 1))
            else:
                cpu_ids.append(int(part))

        return cpu_ids


def set_cpu_affinity_for_gpu(gpu_idx: int):
    def get_gpu_pci_bus(gpu_idx: int) -> PCIBus:
        try:
            cmd = [
                "nvidia-smi", "--query-gpu=pci.domain,pci.bus",
                f"-i={gpu_idx}", "--format=csv,noheader"
            ]
            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True).stdout.strip()

            domain, number = map(lambda x: x.strip(), result.split(','))

            return PCIBus(gpu_idx, domain, number)
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
            return None

    pci_bus = get_gpu_pci_bus(gpu_idx)
    cpu_ids = pci_bus.cpu_affinity_ids()

    if cpu_ids is None:
        return

    os.sched_setaffinity(os.getpid(), cpu_ids)

    logger.info(
        f"Set CPU affinity for GPU {gpu_idx}: {pci_bus.cpu_affinity()}")


def init_distributed(backend="nccl", ) -> None:
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


def deinit_distributed() -> None:
    dist.destroy_process_group()
    destroy_model_parallel()
