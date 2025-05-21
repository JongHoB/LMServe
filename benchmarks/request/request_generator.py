from typing import List, Tuple, Union, Dict

from datasets import load_dataset
from transformers import AutoTokenizer

from .request import APIRequest


# dataset_name: ('subset', 'split', 'prompt', 'output')
dataset_cols = {
    "tatsu-lab/alpaca": (None, "train", "instruction", "output"),
    "openai/openai_humaneval": (None, "test", "prompt", "canonical_solution"),
}


def load_and_preprocess_dataset(
    dataset_name: str,
    tokenizer_name: str,
    max_length: int,
):
    if dataset_name in dataset_cols.keys():
        subset, split, prompt_col, output_col = dataset_cols[dataset_name]
    else:
        raise RuntimeError(
            f"The dataset '{dataset_name}' does not specify which column to"
            "tokenize. If you want to use this dataset, please define"
            "'dataset_cols' in 'request_generator.py'"
        )

    dataset = load_dataset(dataset_name, subset, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(data):
        tokenized_input = tokenizer(data[prompt_col],
                                    truncation=True,
                                    max_length=max_length)
        tokenized_output = tokenizer(data[output_col],
                                     truncation=True,
                                     max_length=max_length)

        input_ids = tokenized_input["input_ids"]
        output_ids = tokenized_output["input_ids"]

        return {
            "prompt": data[prompt_col],
            "output": data[output_col],
            "input_ids": input_ids,
            "output_ids": output_ids,
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def generate_requests(
    dataset_name: str,
    tokenizer_name: str,
    max_seq_len: int,
    num_requests: int,
    num_samples: int,
) -> Union[List[APIRequest], Tuple[List[APIRequest], Dict[str, str]]]:

    dataset = load_and_preprocess_dataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_length=max_seq_len,
    )

    requests: List[APIRequest] = []
    num_requests_counter = 0
    while num_requests_counter < num_requests:
        dataset = dataset.shuffle()
        for data in dataset.to_iterable_dataset():
            input_len = len(data['input_ids'])
            output_len = len(data['output_ids'])
            if (input_len + output_len) > max_seq_len:
                continue

            request = APIRequest(
                prompt=data['prompt'],
                num_samples=num_samples,
                max_output_len=output_len,
            )
            requests.append(request)

            num_requests_counter += 1

            if num_requests_counter >= num_requests:
                break

    return requests
