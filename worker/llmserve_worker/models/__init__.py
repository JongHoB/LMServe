import torch

from typing import Optional

from llmserve_worker.models.opt import OPTForCausalLM
from llmserve_worker.models.llama import LlamaForCausalLM
from llmserve_worker.models.qwen2 import Qwen2ForCausalLM
from llmserve_worker.models.utils import hf_weight_iter
from transformers import AutoConfig


class ModelConfig:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

        tf_config = AutoConfig.from_pretrained(self.model_name,
                                               trust_remote_code=False)
        self.model_type = tf_config.model_type
        self.num_layers = tf_config.num_hidden_layers
        self.hidden_size = tf_config.hidden_size
        self.num_heads = tf_config.num_attention_heads
        self.num_kv_heads = getattr(tf_config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.dtype = getattr(tf_config, "torch_dtype", torch.float16)
        self.tf_config = tf_config


def get_model(
    model_config: ModelConfig,
    max_input_tokens: Optional[int] = None,
):
    model_type = model_config.model_type.lower()
    dtype = model_config.dtype

    torch.set_default_dtype(dtype)
    with torch.device('cuda'):
        if model_type == "opt":
            model = OPTForCausalLM(model_config.tf_config)
        elif model_type == "llama":
            model = LlamaForCausalLM(model_config.tf_config)
        elif model_type == "qwen2":
            model = Qwen2ForCausalLM(model_config.tf_config)
        else:
            raise ValueError(f"Unspported model type {model_type}")

    model.load_weights(hf_weight_iter(model_config.model_name))

    return model.eval()
