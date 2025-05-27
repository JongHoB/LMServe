import torch
import torch.nn as nn
import math

from typing import Tuple

from llmserve_worker.ops.rotary_embedding_ops import rotary_pos_emb


class RotaryEmbedding(nn.Module):
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

        # Force float32
        # (see https://github.com/huggingface/transformers/pull/29285)
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
