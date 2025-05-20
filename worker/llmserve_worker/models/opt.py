import torch
from torch import nn

from typing import List, Tuple, Iterable, Optional

from llmserve_worker.models.attention import Attention
from llmserve_worker.models.activations import ACT2FN
from llmserve_worker.models.input_params import InputParams


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

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                "hidden_size must be divisible by num heads"
                f"(got 'hidden_size': {hidden_size} and 'num_heads': "
                f"{num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

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
        self.fc1 = nn.Linear(
            self.hidden_size,
            config.ffn_dim,
            bias=config.enable_bias,
        )
        self.fc2 = nn.Linear(
            config.ffn_dim,
            self.hidden_size,
            bias=config.enable_bias,
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

        self.embed_tokens = nn.Embedding(
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

        self.lm_head = self.model.decoder.embed_tokens.weight

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

        logits = torch.matmul(hidden_states, self.lm_head.t())
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "lm_head.weight" in name and self.config.tie_word_embeddings:
                continue
            if name.startswith("decoder."):
                name = "model." + name

            params_dict[name].data.copy_(loaded_weight)
