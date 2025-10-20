import json
import os
import subprocess
import pandas as pd
import numpy as np

from loguru import logger
from datasets import Dataset
from transformers import AutoTokenizer

from .generator_utils import get_tokenizer_type
from .dataset_config import get_dataset_config, DatasetConfig


base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_names = ["sharegpt", "azure_conv", "azure_code"]


def check_and_make_data_path(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)


def download_dataset(url: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Download dataset...")
    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", output_path, url],
        check=True,
    )


def prepare_sharegpt():
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    data_path = os.path.join(base_dir, "data")
    dataset_path = os.path.join(data_path,
                                "ShareGPT_V3_unfiltered_cleaned_split.json")

    download_dataset(url, dataset_path)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        convs = item.get("conversations", [])
        human_msg = ""
        gpt_msg = ""
        for conv in convs:
            if conv.get("from") == "human" and not human_msg:
                human_msg = conv.get("value", "")
            elif conv.get("from") == "gpt" and not gpt_msg:
                gpt_msg = conv.get("value", "")
            if human_msg and gpt_msg:
                break
        else:
            continue

        data.append({"id": item.get("id"), "human": human_msg, "gpt": gpt_msg})

    dataset = Dataset.from_list(data)

    output_path = os.path.join(base_dir, "datasets/sharegpt")
    dataset.save_to_disk(output_path)


def prepare_azure(tokenizer_name: str, type: str):
    url = f"https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_{type}.csv"

    data_path = os.path.join(base_dir, "data")
    dataset_path = os.path.join(data_path,
                                f"AzureLLMInferenceTrace_{type}.csv")

    download_dataset(url, dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_type = get_tokenizer_type(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    timestamps = []
    input_ids = []
    output_ids = []

    df = pd.read_csv(
        dataset_path,
        names=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
        header=0,
        parse_dates=["TIMESTAMP"],
    )

    start = df["TIMESTAMP"].iloc[0]
    df["TIMESTAMP"] = (df["TIMESTAMP"] - start).dt.total_seconds()
    for _, row in df.iterrows():
        timestamp = row["TIMESTAMP"]
        input_len = row["ContextTokens"].astype(int)
        output_len = row["GeneratedTokens"].astype(int)

        timestamps.append(timestamp)
        input_ids.append(np.random.randint(vocab_size, size=input_len))
        output_ids.append(np.random.randint(vocab_size, size=output_len))

    inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    data = [{
        "timestamp": timestamp,
        "input": in_text,
        "output": out_text
    } for timestamp, in_text, out_text in zip(timestamps, inputs, outputs)]

    dataset = Dataset.from_list(data)

    output_path = os.path.join(base_dir,
                               f"datasets/azure_{type}_{tokenizer_type}")
    dataset.save_to_disk(output_path)


def prepare_dataset(dataset: str, tokenizer: str):
    if dataset == "sharegpt":
        prepare_sharegpt()
    elif dataset == "azure_conv":
        prepare_azure(tokenizer, "conv")
    elif dataset == "azure_code":
        prepare_azure(tokenizer, "code")

    config: DatasetConfig = get_dataset_config(dataset, tokenizer)
    return config.load_dataset(tokenizer)
