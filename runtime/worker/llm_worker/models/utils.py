import torch
import os
import re

from typing import List, Optional, Tuple

from huggingface_hub import hf_api, snapshot_download
from safetensors.torch import load_file as load_safetensors


def get_layer_type(model, param_name):
    match = re.match(r"^(.*)\.(weight|bias)$", param_name)
    if not match:
        raise ValueError(f"Invalid param_name format: {param_name}")

    layer_name, param_type = match.groups()

    def rgetattr(obj, attr):
        for name in attr.split('.'):
            obj = getattr(obj, name)
        return obj

    layer_type = type(rgetattr(model, layer_name)).__name__

    return layer_type, param_type


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


def load_weights(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    layer_types: Tuple[str, str],
    tensor_model_parallel_rank: int,
) -> None:
    layer_type, param_type = layer_types

    if layer_type in ["VocabParallelEmbedding", "ColumnParallelLinear"]:
        shard_size = param.shape[0]
        start_idx = tensor_model_parallel_rank * shard_size
        end_idx = (tensor_model_parallel_rank + 1) * shard_size
        loaded_weight = loaded_weight[start_idx:end_idx]

    elif layer_type in ["RowParallelLinear"]:
        if param_type == "bias":
            shard_size = param.shape[0]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            loaded_weight = loaded_weight[:, start_idx:end_idx]
        else:
            shard_size = param.shape[1]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            loaded_weight = loaded_weight[start_idx:end_idx]

    assert param.shape == loaded_weight.shape, (
        f"Tensor shape mismatch between model and checkpoint: "
        f"{param.shape} != {loaded_weight.shape}")
    param.data.copy_(loaded_weight)


def travel_layers(
    mod: torch.nn.Module,
    includes: Optional[List[str]] = None,
):
    layers = []
    if len(list(mod.children())) == 0:
        if includes is not None:
            for include in includes:
                if include.lower() in mod.__class__.__name__.lower():
                    return [mod]
            else:
                return []

        return [mod]
    else:
        for i, (name, child) in enumerate(mod.named_children()):
            layers += travel_layers(child, includes)

        return layers
