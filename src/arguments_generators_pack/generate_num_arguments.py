from enum import Enum
from functools import lru_cache

import numpy as np

from llm_sdk import Small_LLM_Model

from .utils import get_vocab_list, softmax


class STATE(Enum):
    START_NUMS = 1
    AFTER_MINUS = 2
    JUST_NUMBERS = 3
    END_NUMS = 4


@lru_cache(maxsize=None)
def get_number_token_ids(
    small_llm: "Small_LLM_Model",
) -> tuple[list[int], list[int]]:
    """Return (digit token ids, null token ids), computed once per model"""
    vocab = get_vocab_list(small_llm)
    digit_allowed_ids = []
    null_ids = [id for token, id in vocab.items() if token.strip() == "null"]
    for token, id in vocab.items():
        clean_token = token.replace(" ", "").replace("Ġ", "").strip()
        if not clean_token:
            continue
        if all(c in " 0123456789." for c in clean_token):
            digit_allowed_ids.append(id)
    return digit_allowed_ids, null_ids


def number_generate(
    small_llm: "Small_LLM_Model",
    prompt_tokens: list[int],
    name_param: str,
    is_last: bool,
) -> list[int] | None:

    res = []

    formatted_name = f'"{name_param}":'

    name_tokens = [t.item() for t in small_llm.encode(formatted_name)[0]]
    res.extend(name_tokens)
    prompt_tokens.extend(name_tokens)
    term = "}" if is_last else ","
    try:
        vocab = get_vocab_list(small_llm)
    except OSError:
        return None

    term_id = vocab.get(term)
    minus_id = vocab.get("-")
    digit_allowed_ids, null_ids = get_number_token_ids(small_llm)

    state = STATE.START_NUMS
    while state != STATE.END_NUMS:
        allowed_tokenids: list[int] = []
        if state == STATE.START_NUMS:
            allowed_tokenids = digit_allowed_ids + [minus_id]  # type: ignore
            allowed_tokenids.extend(null_ids)
        elif state == STATE.AFTER_MINUS or state == STATE.JUST_NUMBERS:
            allowed_tokenids = digit_allowed_ids + [term_id]  # type: ignore
            allowed_tokenids.append(minus_id)  # type: ignore

        logits = small_llm.get_logits_from_input_ids(prompt_tokens)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_tokenids] = 0
        masked_logits = logits + mask
        next_token_id = int(np.argmax(softmax(masked_logits)))
        prompt_tokens.append(next_token_id)
        res.append(next_token_id)

        if state == STATE.START_NUMS:
            if next_token_id == minus_id:
                state = STATE.AFTER_MINUS
            else:
                state = STATE.JUST_NUMBERS
        elif state == STATE.AFTER_MINUS:
            state = STATE.JUST_NUMBERS
        elif state == STATE.JUST_NUMBERS:
            if next_token_id == term_id:
                state = STATE.END_NUMS
    return res
