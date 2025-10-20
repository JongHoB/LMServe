import os

from typing import Optional
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk

from .generator_utils import get_tokenizer_type


@dataclass
class DatasetConfig:
    prompt_col: str
    output_col: str
    path: Optional[str] = None
    dataset_path: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    with_tokenizer: bool = False

    def load_dataset(self, tokenizer_name: str, **kwargs):
        if self.dataset_path:
            if self.with_tokenizer:
                tokenizer_type = get_tokenizer_type(tokenizer_name)
                dataset_path = f"{self.dataset_path}_{tokenizer_type}"
            else:
                dataset_path = self.dataset_path

            return load_from_disk(dataset_path, **kwargs)

        elif self.path:
            return load_dataset(self.path, self.subset, split=self.split,
                                **kwargs)

        else:
            raise RuntimeError("Dataset source is undefined: "
                               "either 'path' or 'dataset_path' must be set")


base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_configs = {
    "alpaca":
    DatasetConfig(
        "instruction",
        "output",
        path="tatsu-lab/alpaca",
        split="train",
    ),
    "humaneval":
    DatasetConfig(
        "prompt",
        "canonical_solution",
        path="openai/openai_humaneval",
        split="test",
    ),
    "sharegpt":
    DatasetConfig(
        "human",
        "gpt",
        dataset_path=os.path.join(base_dir, "datasets/sharegpt"),
    ),
    "azure_conv":
    DatasetConfig(
        "input",
        "output",
        dataset_path=os.path.join(base_dir, "datasets/azure_conv"),
        with_tokenizer=True,
    ),
    "azure_code":
    DatasetConfig(
        "input",
        "output",
        dataset_path=os.path.join(base_dir, "datasets/azure_code"),
        with_tokenizer=True,
    ),
}


def get_dataset_config(
    dataset_name: str,
    tokenizer_name: str,
) -> DatasetConfig:
    if dataset_name not in dataset_configs.keys():
        raise RuntimeError(
            f"The dataset '{dataset_name}' does not specify which column to"
            "tokenize. If you want to use this dataset, please define"
            "'dataset_cols' in 'request_generator.py'")

    config = dataset_configs[dataset_name]

    return config
