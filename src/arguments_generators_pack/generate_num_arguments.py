import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_sdk import Small_LLM_Model
from enum import Enum
import json
from typing import Any


class STATE(Enum):
    START_NUMS = 1
    AFTER_MINUS = 2
    JUST_NUMBERS = 3
    END_NUMS = 4


def get_vocab_list(small_llm: "Small_LLM_Model"):
    """Get vocab list from vocab.json
    """
    json_path = small_llm.get_path_to_vocab_file()
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    res = []
    return data


def softmax(x: Any):
    """softmax"""
    x_max = np.max(x, keepdims=True)
    exp_x = np.exp(x - x_max)

    return exp_x / np.sum(exp_x,keepdims=True)


def number_generate(
    small_llm: "Small_LLM_Model",
    promt_tokenst: list[int],
    name_param: str,
    is_last: bool
) -> list[int]:
    has_dot: bool = False

    res = []

    formatted_name = f'"{name_param}":'

    digits = [str(d) for d in range(0, 10)]
    name_tokens = [t.item() for t in small_llm.encode(formatted_name)[0]]
    res.extend(name_tokens)
    promt_tokenst.extend(name_tokens)
    term = "}" if is_last else ","
    vocab = get_vocab_list(small_llm)
    term_id = vocab.get(term)
    minus_id = vocab.get("-")
    digit_allowed_ids = []
    null_ids = [id for token, id in vocab.items() if token.strip() == "null"]
    for token, id in vocab.items():
        clean_token = token.replace(" ", "").replace("Ġ", "").strip()
        if not clean_token:
            continue
        if all(c in " 0123456789" for c in clean_token):
            digit_allowed_ids.append(id)

    state = STATE.START_NUMS
    while state != STATE.END_NUMS:
        allowed_tokenids = []
        if state == STATE.START_NUMS:
            allowed_tokenids = digit_allowed_ids + [minus_id]
            allowed_tokenids.extend(null_ids)
        elif state == STATE.AFTER_MINUS or state == STATE.JUST_NUMBERS:
            allowed_tokenids = digit_allowed_ids + [term_id, minus_id]

        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_tokenids] = 0
        masked_logits = logits + mask
        next_token_id = int(np.argmax(softmax(masked_logits)))
        promt_tokenst.append(next_token_id)
        res.append(next_token_id)

        decoded = small_llm.decode(next_token_id)

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
