import json
import csv
import os
import argparse
import subprocess
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_names = ["sharegpt", "azure"]


def check_and_make_data_path(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)


def prepare_sharegpt():
    data_path = os.path.join(base_dir, "data")
    check_and_make_data_path(data_path)

    dataset_path = os.path.join(data_path,
                                "ShareGPT_V3_unfiltered_cleaned_split.json")

    if not os.path.isfile(dataset_path):
        print("Download dataset...")
        subprocess.run([
            "wget", "-O", dataset_path,
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        ])

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


def prepare_azure(tokenizer_name: AutoTokenizer):
    data_path = os.path.join(base_dir, "data")
    check_and_make_data_path(data_path)

    dataset_path = os.path.join(data_path, "AzureLLMInferenceTrace_conv.csv")

    if not os.path.isfile(dataset_path):
        print("Download dataset...")
        subprocess.run([
            "wget", "-O", dataset_path,
            "https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_conv.csv"
        ])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    input_ids = []
    output_ids = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        reader.__next__()
        for timestamp, input_len, output_len in reader:
            input_len = int(input_len)
            output_len = int(output_len)
            input_ids.append(np.random.randint(vocab_size, size=input_len))
            output_ids.append(np.random.randint(vocab_size, size=output_len))

    inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    data = [{
        "input": in_text,
        "output": out_text
    } for in_text, out_text in zip(inputs, outputs)]

    dataset = Dataset.from_list(data)

    output_path = os.path.join(base_dir, "datasets/azure")
    dataset.save_to_disk(output_path)


def prepare_dataset(dataset: str, tokenizer: str):
    if dataset == "sharegpt":
        prepare_sharegpt()
    elif dataset == "azure":
        if not tokenizer:
            raise ValueError(
                "'--tokenizer' argument is required because the dataset is generated based on the provided tokenizer"
            )
        prepare_azure(tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        choices=dataset_names,
                        default="sharegpt")
    parser.add_argument("--tokenizer", type=str)
    args = parser.parse_args()

    prepare_dataset(dataset=args.dataset, tokenizer=args.tokenizer)
    prepare_sharegpt()
