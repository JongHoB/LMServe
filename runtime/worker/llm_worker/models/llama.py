# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
# Copyright 2025 The Korea University CSL team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from typing import List, Tuple, Iterable, Optional

from llm_worker.models.parallel import (VocabParallelEmbedding,
                                        ColumnParallelLinear,
                                        RowParallelLinear)
from llm_worker.models.parallel import (get_model_parallel_rank,
                                        get_model_parallel_world_size)
from llm_worker.models.utils import get_layer_type, load_weights
from llm_worker.models.input_params import InputParams
from llm_worker.models.layers.activations import ACT2FN
from llm_worker.models.layers.attention import Attention
from llm_worker.models.layers import RotaryEmbedding


class LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaMLP(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        down_proj = self.down_proj(
            self.activation_fn(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))
        return down_proj


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        tp_size = get_model_parallel_world_size()

        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        assert num_heads % tp_size == 0
        assert num_kv_heads % tp_size == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = hidden_size // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.q_size,
            bias=False,
            gather_output=False,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.kv_size,
            bias=False,
            gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.kv_size,
            bias=False,
            gather_output=False,
        )
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
        )

        self.rotary_emb = RotaryEmbedding(config=config)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scaling,
            max_position_ids=config.max_position_embeddings,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_params: InputParams,
        kv_cache: Optional[Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k = self.rotary_emb(
            q,
            k,
            input_params.position_ids,
        )
        attn_output = self.attn(
            q,
            k,
            v,
            kv_cache=kv_cache,
            input_params=input_params,
        )

        output = self.o_proj(attn_output)

        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)

        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_params: InputParams,
        kv_cache: Optional[Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            input_params=input_params,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_params: InputParams,
        kv_caches: Optional[List[Tuple[torch.Tensor]]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                input_params=input_params,
                kv_cache=(kv_caches[i] if kv_caches is not None else None),
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_params: InputParams,
        kv_caches: Optional[List[Tuple[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            input_params=input_params,
            kv_caches=kv_caches,
        )
        cu_output_lens = input_params.cu_seqlens_q[1:]
        indices = cu_output_lens - 1
        hidden_states = hidden_states[indices]
        logits = self.lm_head(hidden_states)

        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        tensor_model_parallel_rank = get_model_parallel_rank()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            load_weights(
                params_dict[name],
                loaded_weight,
                get_layer_type(self, name),
                tensor_model_parallel_rank,
            )

        if self.config.tie_word_embeddings:
            self.lm_head.weight.data.copy_(self.model.embed_tokens.weight)
