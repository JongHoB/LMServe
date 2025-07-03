import torch
import flashinfer

from typing import Tuple


def _max_fn(x):
    """
        max(x, 0))
    """
    return torch.maximum(x, torch.zeros_like(x))


def _min_fn(x):
    """
        min(x, 1))
    """
    return torch.minimum(x, torch.ones_like(x))


def sample(
    logits: torch.Tensor,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_probs = logits.softmax(-1)
    if top_k != 1 and top_p != 0.0 and temperature != 0.0:
        output_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits.view(-1, logits.size(-1)) / temperature,
            top_k=top_k,
            top_p=top_p,
        ).to(torch.int64).view(logits.shape[:-1])
        probs = torch.gather(all_probs, -1,
                             output_ids.unsqueeze(-1)).squeeze(-1)
    else:
        probs, output_ids = torch.max(all_probs, dim=-1)

    return output_ids, probs, all_probs
