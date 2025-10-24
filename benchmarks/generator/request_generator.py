import os
import numpy as np

from loguru import logger
from datasets import Dataset

from typing import List, Optional, Tuple
from transformers import AutoTokenizer

from .request import APIRequest, APIRequestSequence
from .dataset_config import get_dataset_config, DatasetConfig
from .prepare_dataset import prepare_dataset


def load_and_preprocess_dataset(
    dataset_name: str,
    tokenizer_name: str,
    max_length: Optional[int] = None,
    include_time: bool = False,
) -> Dataset:
    config: DatasetConfig = get_dataset_config(dataset_name, tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    try:
        dataset = config.load_dataset(tokenizer_name)
    except FileNotFoundError as e:
        logger.warning(f"Dataset not found: {e}")
        logger.info("Preparing dataset...")

        dataset = prepare_dataset(dataset_name, tokenizer_name)

    if include_time:
        if "timestamp" not in dataset.column_names:
            raise RuntimeError(
                f"Dataset '{dataset_name}' does not have a 'timestamp' column")

    def tokenize_function(data):
        tokenized_input = tokenizer(data[config.prompt_col],
                                    truncation=True,
                                    max_length=max_length)
        tokenized_output = tokenizer(data[config.output_col],
                                     truncation=True,
                                     max_length=max_length)

        input_ids = tokenized_input["input_ids"]
        output_ids = tokenized_output["input_ids"]

        ret = {
            "prompt": data[config.prompt_col],
            "output": data[config.output_col],
            "input_ids": input_ids,
            "output_ids": output_ids,
        }

        if include_time:
            ret["timestamp"] = data["timestamp"]

        return ret

    dataset = dataset.shuffle()

    allow_batch = (not config.multi_turn)
    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=allow_batch,
                                    num_proc=os.cpu_count())
    return tokenized_dataset


def generate_requests(
    dataset_name: str,
    tokenizer_name: str,
    max_seq_len: int,
    num_requests: int,
    num_samples: int,
    ignore_eos: bool,
    disable_cache: bool,
) -> List[APIRequest]:

    dataset = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_length=max_seq_len,
    )

    requests: List[APIRequest] = []
    while len(requests) < num_requests:
        for data in dataset.to_iterable_dataset():
            input_len = len(data['input_ids'])
            output_len = len(data['output_ids'])
            if (output_len <= 4) or ((input_len + output_len) > max_seq_len):
                continue

            request = APIRequest(
                prompt=data['prompt'],
                num_samples=num_samples,
                max_output_len=output_len,
                ignore_eos=ignore_eos,
                disable_cache=disable_cache,
            )
            requests.append(request)
            if len(requests) >= num_requests:
                break

    return requests


def generate_trace(
    dataset_name: str,
    tokenizer_name: str,
    num_requests: int,
    disable_cache: bool,
) -> Tuple[List[APIRequest], List[float]]:

    dataset = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        include_time=True,
    )

    requests: List[APIRequest] = []
    timestamps: List[float] = []
    for data in dataset.to_iterable_dataset():
        request = APIRequest(
            prompt=data['prompt'],
            num_samples=1,
            max_output_len=len(data['output_ids']),
            ignore_eos=True,
            disable_cache=disable_cache,
        )

        requests.append(request)
        timestamps.append(data['timestamp'])
        if len(requests) >= num_requests:
            break

    intervals = [0] + [(timestamps[i] - timestamps[i - 1])
                       for i in range(1, len(timestamps))]

    return (requests, intervals)


def generate_radom_requests(
    tokenizer_name: str,
    max_input_len: int,
    max_output_len: int,
    max_seq_len: int,
    num_requests: int,
    num_samples: int,
    disable_cache: bool,
) -> List[APIRequest]:

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    requests: List[APIRequest] = []
    while len(requests) < num_requests:
        input_len = np.random.randint(min(max_input_len, 4), max_input_len)
        output_len = np.random.randint(min(max_output_len, 32), max_output_len)
        if (input_len + output_len) > max_seq_len:
            continue

        input_ids = np.random.randint(vocab_size, size=input_len)
        prompt = tokenizer.decode(input_ids)

        request = APIRequest(
            prompt=prompt,
            num_samples=num_samples,
            max_output_len=output_len,
            ignore_eos=True,
            disable_cache=disable_cache,
        )
        requests.append(request)

    return requests


def generate_chat_requests(
    dataset_name: str,
    tokenizer_name: str,
    num_sessions: int,
    num_samples: int,
    ignore_eos: bool,
    disable_cache: bool,
) -> List[APIRequestSequence]:

    dataset = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
    )

    req_seqs: List[APIRequestSequence] = []
    for i, data in enumerate(dataset.to_iterable_dataset()):
        requests: List[APIRequest] = []
        output_texts: List[str] = []
        for prompt, output, output_ids in zip(data['prompt'],
                                              data['output'],
                                              data['output_ids']):
            request = APIRequest(
                prompt=prompt,
                num_samples=num_samples,
                max_output_len=len(output_ids),
                ignore_eos=ignore_eos,
                disable_cache=disable_cache,
            )
            requests.append(request)
            output_texts.append(output)

        req_seq = APIRequestSequence(requests=requests,
                                     output_texts=output_texts)
        req_seqs.append(req_seq)
        if len(req_seqs) >= num_sessions:
            break

    return req_seqs
