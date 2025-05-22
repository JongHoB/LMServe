import torch
import os

from typing import List

from huggingface_hub import hf_api, snapshot_download
from safetensors.torch import load_file as load_safetensors


def _get_hf_weight_files(
    model_name: str,
    extensions: List[str] = [".bin", ".safetensors"],
):
    """
    Get the weight files from the Hugging Face hub for the given model.

    Supports both .bin and .safetensors extensions.
    """
    repo_files = hf_api.list_repo_files(model_name, repo_type="model")

    weight_files = []
    for ext in extensions:
        weight_files = [f for f in repo_files if f.endswith(ext)]
        if len(weight_files) > 0:
            break

    if len(weight_files) == 0:
        raise FileNotFoundError(f"No weight files found in {model_name}")

    allow_patterns = [f"*{ext}" for ext in extensions]
    try:
        local_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
        for fname in weight_files:
            if not os.path.exists(os.path.join(local_path, fname)):
                raise FileNotFoundError

    except FileNotFoundError:
        local_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
        )

    files: List[str] = []
    for fname in weight_files:
        files.append(os.path.join(local_path, fname))

    return files


def hf_weight_iter(model_name: str):
    """
    Iterate over the weights of the Hugging Face model.

    Supports loading .bin files with PyTorch and .safetensors files with
    safetensors.
    """
    hf_weight_files = _get_hf_weight_files(model_name)

    for file in hf_weight_files:
        if file.endswith(".safetensors"):
            state = load_safetensors(file, device="cpu")
        else:
            state = torch.load(file, weights_only=True, map_location="cpu")

        for name, param in state.items():
            yield name, param
