import triton
import triton.language as tl


@triton.jit
def rejection_sample_kernel(target_token_ids_ptr, drfat_token_ids_ptr,
                            target_probs_ptr, draft_probs_ptr,
                            target_all_probs, draft_all_probs, batch_size,
                            num_draft_tokens, vocab_size,
                            BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    target_token_ids_start = target_token_ids_ptr + block_idx * (
        num_draft_tokens + 1)
    draft_token_ids_start = draft_token_ids_start + block_idx * num_draft_tokens
    target_probs_start = taget_probs_ptr + block_idx * num_draft_tokens
    draft_probs_start = draft_probs_ptr + block_idx * num_draft_tokens
    target_all_probs_start = taget_all_probs_ptr + block_idx * num_draft_tokens * vocab_size
    draft_all_probs_start = draft_all_probs_ptr + block_idx * num_draft_tokens * vocab_size

