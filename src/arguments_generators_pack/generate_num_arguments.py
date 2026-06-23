import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_sdk import Small_LLM_Model
from enum import Enum
import json


class STATE(Enum):
    START_NUMS = 1
    AFTER_MINUS = 2
    JUST_NUMBERS = 3
    END_NUMS = 4

def get_vocab_list(small_llm:"Small_LLM_Model"):
    
    json_path = small_llm.get_path_to_vocab_file()
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    res = []
    return data


def number_generate(small_llm:"Small_LLM_Model",promt_tokenst:list[int],name_param:str, is_last:bool,user_request:str) -> list[int]:
    has_dot:bool = False
    
    res = []
    
    formatted_name = f'\"{name_param}\":'

    digits = [str(d) for d in range(0,10)]
    name_tokens = [t.item() for t in small_llm.encode(formatted_name)[0]]
    res.extend(name_tokens)
    promt_tokenst.extend(name_tokens)
    term = '}' if is_last else ','
    vocab = get_vocab_list(small_llm)
    digit_allowed_ids = [id for token, id in vocab.items() 
                     if all(c in '0123456789.' for c in token)]
    quote_id = vocab['"']

    state = STATE.START_NUMS
    while state != STATE.END_NUMS:
        allowed_tokenids = []
        if state == STATE.START_NUMS:
            allowed_chars = ['-'] + digits
        elif state == STATE.AFTER_MINUS or state == STATE.JUST_NUMBERS:
            allowed_chars = digits + [term]
    
        allowed_tokenids = [small_llm.encode(s)[0][0].item() for s in allowed_chars]

        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_tokenids] = 0
        masked_logits = logits + mask

        next_token_id = int(np.argmax(masked_logits))

        promt_tokenst.append(next_token_id)
        res.append(next_token_id)

        decoded = small_llm.decode(next_token_id)
        print(decoded)
        if small_llm.decode(next_token_id) == ".":
            has_dot = True
        if state == STATE.START_NUMS:
            if small_llm.decode(next_token_id) == '-':
                state = STATE.AFTER_MINUS
            else:
                state = STATE.JUST_NUMBERS
        elif state == STATE.AFTER_MINUS:
            state = STATE.JUST_NUMBERS
        elif state == STATE.JUST_NUMBERS:
            if  ',' in decoded:
                state = STATE.END_NUMS
            elif '}' in decoded:
                state = STATE.END_NUMS
    return res
        
        
