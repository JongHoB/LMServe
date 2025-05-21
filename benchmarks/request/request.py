from typing import List, TypedDict, Optional


class APIRequest(TypedDict):
    session_id: Optional[str]
    prompt: str
    num_samples: int
    max_output_len: Optional[int]


class APIResponse(TypedDict):
    token_ids: List[int]
    output_len: int
