import torch
from torch import nn

from typing import List, Tuple, Iterable, Optional

from llmserve_worker.models.parallel import (VocabParallelEmbedding,
                                             ColumnParallelLinear,
                                             RowParallelLinear)
from llmserve_worker.models.parallel import (get_model_parallel_rank,
                                             get_model_parallel_world_size)
from llmserve_worker.models.utils import get_layer_type, load_weights
from llmserve_worker.models.input_params import InputParams
from llmserve_worker.models.layers.activations import ACT2FN
from llmserve_worker.models.layers.attention import Attention


class OPTPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(self, config, bias: bool = True):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        tp_size = get_model_parallel_world_size()

        assert hidden_size % num_heads == 0
        assert num_heads % tp_size == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads // tp_size
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                "hidden_size must be divisible by num heads"
                f"(got 'hidden_size': {hidden_size} and 'num_heads': "
                f"{num_heads}).")
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

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scaling,
            max_position_ids=config.max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_params: InputParams,
        kv_cache: Optional[Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        attn_output = self.attn(
            q,
            k,
            v,
            input_params=input_params,
            kv_cache=kv_cache,
        )
        attn_output = self.out_proj(attn_output)

        return attn_output


class OPTDecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OPTAttention(config, bias=config.enable_bias)
        self.do_layer_norm_before = config.do_layer_norm_before
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.hidden_size,
            elementwise_affine=config.layer_norm_elementwise_affine,
        )
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            gather_output=False,
        )
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            input_is_parallel=True,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.hidden_size,
            elementwise_affine=config.layer_norm_elementwise_affine,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_params: InputParams,
        kv_cache: Optional[Tuple[torch.Tensor]],
    ) -> torch.Tensor:

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_params=input_params,
        )
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        self.embed_positions = OPTPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size,
                config.word_embed_proj_dim,
                bias=False,
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim,
                config.hidden_size,
                bias=False,
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been
        # fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: torch.Tensor,
        input_params: InputParams,
        kv_caches: Optional[List[Tuple[torch.Tensor]]],
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(input_params.position_ids)

        if self.project_in is not None:
            input_embeds = self.project_in(input_embeds)

        hidden_states = input_embeds + pos_embeds

        # decoder layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                input_params=input_params,
                kv_cache=(kv_caches[i] if kv_caches is not None else None),
            )

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.decoder = OPTDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_params: InputParams,
        kv_caches: Optional[List[Tuple[torch.Tensor]]],
    ) -> torch.Tensor:
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            input_params=input_params,
            kv_caches=kv_caches,
        )
        return decoder_outputs


class OPTForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = OPTModel(config)

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
        hidden_states = self.model.decoder(
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
            if "lm_head.weight" in name and self.config.tie_word_embeddings:
                continue

            if name.startswith("decoder."):
                name = "model." + name

            load_weights(
                params_dict[name],
                loaded_weight,
                get_layer_type(self, name),
                tensor_model_parallel_rank,
            )

        self.lm_head.weight.data.copy_(self.model.embed_tokens.weight)
