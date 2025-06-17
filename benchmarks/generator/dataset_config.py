import os

from typing import Dict, Any
from collections.abc import Callable
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk


@dataclass
class DatasetConfig:
    load_fn: Callable
    args: Dict[str, Any]
    prompt_col: str
    output_col: str


base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_configs = {
    "alpaca":
    DatasetConfig(load_dataset, {
        "path": "tatsu-lab/alpaca",
        "subset": None,
        "split": "train"
    }, "instruction", "output"),
    "humaneval":
    DatasetConfig(load_dataset, {
        "path": "openai/openai_humaneval",
        "subset": None,
        "split": "test"
    }, "prompt", "canonical_solution"),
    "sharegpt":
    DatasetConfig(load_from_disk, {
        "dataset_path": os.path.join(base_dir, "datasets/sharegpt"),
    }, "human", "gpt"),
    "azure":
    DatasetConfig(load_from_disk, {
        "dataset_path": os.path.join(base_dir, "datasets/azure"),
    }, "input", "output"),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name in dataset_configs.keys():
        config = dataset_configs[dataset_name]
    else:
        raise RuntimeError(
            f"The dataset '{dataset_name}' does not specify which column to"
            "tokenize. If you want to use this dataset, please define"
            "'dataset_cols' in 'request_generator.py'")

    return config
