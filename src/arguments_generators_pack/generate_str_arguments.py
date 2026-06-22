import os
import sys
import numpy as np
from enum import Enum
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_sdk import Small_LLM_Model
import string

class STATE(Enum):
    START_STR = 0
    AFTER_QUOTES = 1
    JUST_SIMBOLS = 2
    END_STR = 3
    AFTER_STR = 4


def str_generator(small_llm:"Small_LLM_Model",promt_tokenst:list[int],name_param:str, is_last:bool) -> list[int]:
    res = []
    
    formatted_name = f'\"{name_param}\":'
    name_tokens = [t.item() for t in small_llm.encode(formatted_name)[0]]
    res.extend(name_tokens)
    promt_tokenst.extend(name_tokens)
    state = STATE.START_STR

    chars_lsit = (
    list(string.ascii_letters) +
    [str(d) for d in range(10)] +
    list(' .,-_/:;=@()[]{}<>+*?^$\\|&!\"')
    )
    term = '}' if is_last else ','

    while state != STATE.END_STR:
        allowed_tokens = []
        if state == STATE.START_STR:
            allowed_chars = ['"']
        elif state == STATE.AFTER_QUOTES:
            allowed_chars = chars_lsit
        elif state == STATE.END_STR:
            allowed_chars = ['"']
        elif state == STATE.AFTER_STR:
            allowed_chars = [term]
        allowed_tokenids = [small_llm.encode(s)[0][0].item() for s in allowed_chars]

        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_tokenids] = 0
        masked_logits = logits + mask

        next_token_id = int(np.argmax(masked_logits))

        promt_tokenst.append(next_token_id)
        res.append(next_token_id)

        decoded = small_llm.decode(next_token_id)
        if  state == STATE.START_STR:
            state = STATE.AFTER_QUOTES
        elif state == STATE.AFTER_QUOTES:
            if '"' in decoded:
                state = STATE.AFTER_STR
        elif state == STATE.AFTER_STR:
            state = STATE.END_STR
    return res