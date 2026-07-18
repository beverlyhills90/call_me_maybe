from enum import Enum

import numpy as np

from llm_sdk import Small_LLM_Model

from .utils import get_vocab_list


class STATE(Enum):
    START_STR = 0
    JUST_SIMBOLS = 1
    END_STR = 2
    AFTER_STR = 3


def str_generator(
    small_llm: "Small_LLM_Model",
    promt_tokenst: list[int],
    name_param: str,
    is_last: bool,
) -> list[int] | None:
    res = []

    formatted_name = f'"{name_param}":'
    name_tokens = [t.item() for t in small_llm.encode(formatted_name)[0]]
    res.extend(name_tokens)
    promt_tokenst.extend(name_tokens)
    try:
        vocab = get_vocab_list(small_llm)
    except OSError:
        return None
    quote_id = vocab.get('"')

    term = "}" if is_last else ","
    term_id = vocab.get(term)
    string_allowed_ids = [id for token, id in vocab.items()
                          if '"' not in token]

    state = STATE.START_STR
    while state != STATE.END_STR:
        allowed_tokenids = []
        if state == STATE.START_STR:
            allowed_tokenids = [quote_id]
        elif state == STATE.JUST_SIMBOLS:
            allowed_tokenids = string_allowed_ids + [quote_id]  # type: ignore
        elif state == STATE.AFTER_STR:
            allowed_tokenids = [term_id]
        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_tokenids] = 0

        masked_logits = logits + mask
        masked_logits[quote_id] += 10.0

        next_token_id = int(np.argmax(masked_logits))
        promt_tokenst.append(next_token_id)
        res.append(next_token_id)

        if state == STATE.START_STR:
            state = STATE.JUST_SIMBOLS
        elif state == STATE.JUST_SIMBOLS:
            if next_token_id == quote_id:
                state = STATE.AFTER_STR
        elif state == STATE.AFTER_STR:
            state = STATE.END_STR
    return res
