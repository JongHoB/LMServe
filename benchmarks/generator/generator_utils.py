import json

from huggingface_hub import hf_hub_download


def get_tokenizer_type(tokenizer_name: str) -> str:
    config_path = hf_hub_download(repo_id=tokenizer_name,
                                  filename="tokenizer_config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_type = config.get("tokenizer_class")

    return tokenizer_type.lower()
