import os
import sys
import numpy as np
from enum import Enum
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_sdk import Small_LLM_Model
import json
from typing import Any

class STATE(Enum):
    START_STR = 0
    JUST_SIMBOLS = 1
    END_STR = 2
    AFTER_STR = 3


def get_vocab_list(small_llm:"Small_LLM_Model"):
    
    json_path = small_llm.get_path_to_vocab_file()
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    res = []
    return data

def softmax(x:Any):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
    

def str_generator(small_llm:"Small_LLM_Model",promt_tokenst:list[int],name_param:str, is_last:bool,user_request:str) -> list[int]:
    res = []
    
    formatted_name = f'\"{name_param}\":'
    name_tokens = [t.item() for t in small_llm.encode(formatted_name)[0]]
    res.extend(name_tokens)
    promt_tokenst.extend(name_tokens)
    vocab = get_vocab_list(small_llm)
    quote_id = vocab.get('"')

    term = '}' if is_last else ','
    term_id = vocab.get(term)
    string_allowed_ids = [
        id for token, id in vocab.items() 
        if '"' not in token or token == '"'
    ]
    

    state = STATE.START_STR
    while state != STATE.END_STR:

        allowed_tokenids = []
        if state == STATE.START_STR:
            allowed_tokenids = [quote_id]
        elif state == STATE.JUST_SIMBOLS:
            allowed_tokenids = string_allowed_ids + [quote_id]
        elif state == STATE.AFTER_STR:
            allowed_tokenids = [term_id]
        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_tokenids] = 0
        
        masked_logits = logits + mask
        masked_logits[quote_id] += 10.0
        probabilities = softmax(masked_logits)
        
        next_token_id = int(np.argmax(masked_logits))
        actual_chance = probabilities[quote_id]
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