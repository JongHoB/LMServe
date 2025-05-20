# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

import torch
import torch.nn as nn

import math

from typing import List, Tuple, Iterable, Optional

from llmserve_worker.models.attention import Attention
from llmserve_worker.models.activations import ACT2FN
from llmserve_worker.models.input_params import InputParams
from llmserve_worker.ops.rotary_embedding_ops import rotary_pos_emb


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config,
        base: int = 10000,
    ) -> None:
        super().__init__()

        base = getattr(config, "rope_theta", base)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = head_dim

        # Create cos and sin embeddings.
        inv_freq = 1.0 / (base**(
            torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).float() /
            self.rotary_dim))

        if getattr(config, "rope_scaling", None) is not None:
            factor = config.rope_scaling["factor"]
            low_freq_factor = config.rope_scaling["low_freq_factor"]
            high_freq_factor = config.rope_scaling["high_freq_factor"]
            old_context_len = config.rope_scaling[
                "original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            # wavelen < high_freq_wavelen: do nothing
            # wavelen > low_freq_wavelen: divide by factor
            inv_freq = torch.where(wavelen > low_freq_wavelen,
                                   inv_freq / factor, inv_freq)
            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor)
            smoothed_inv_freq = ((1 - smooth_factor) * inv_freq / factor +
                                 smooth_factor * inv_freq)
            is_medium_freq = (~(wavelen < high_freq_wavelen) *
                              ~(wavelen > low_freq_wavelen))
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq)

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = inv_freq.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            t = torch.arange(self.max_position_embeddings).float()
            freqs = torch.einsum("i,j->ij", t, inv_freq.float())
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos.to(dtype=config.torch_dtype)
        sin = sin.to(dtype=config.torch_dtype)
        cos_sin = torch.cat((cos, sin), dim=-1)

        self.register_buffer("cos_sin_cached", cos_sin, persistent=False)

    def forward(
            self,
            query: torch.Tensor,  # [num_tokens, num_heads, head_size]
            key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
            positions: torch.Tensor,  # [num_tokens]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query, key = rotary_pos_emb(
            query,
            key,
            positions,
            self.cos_sin_cached,
            self.head_dim
        )
        return query, key


class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size,
                                   config.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size,
                                 config.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,
                                   config.hidden_size,
                                   bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        down_proj = self.down_proj(
            self.activation_fn(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))
        return down_proj


class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        self.hidden_size = hidden_size
        assert num_heads % num_kv_heads == 0
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
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


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)

        self.input_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
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


class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)

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


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_params: InputParams,
        kv_caches: Optional[List[Tuple[torch.Tensor]]],
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            input_params=input_params,
            kv_caches=kv_caches,
        )
        logits = self.lm_head(hidden_states)

        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            params_dict[name].data.copy_(loaded_weight)

        if self.config.tie_word_embeddings:
            self.lm_head.weight.data.copy_(self.model.embed_tokens.weight)


class Qwen2ForProcessRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = 2
        self.config = config
        self.model = Qwen2Model(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels)
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
        logits = self.score(hidden_states)

        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "lm_head.weight" in name:
                continue
            params_dict[name].data.copy_(loaded_weight)
