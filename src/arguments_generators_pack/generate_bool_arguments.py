import numpy as np

from llm_sdk import Small_LLM_Model
from src.promts import Trie

from .utils import get_vocab_list


def bool_generate(
    small_llm: "Small_LLM_Model",
    promt_tokenst: list[int],
    name_param: str,
    is_last: bool,
) -> list[int] | None:
    """Generation of bool arguments
    ARGS:

    small_llm -
    promt_tokenst -
    name_param -
    is_last - boolean
    user_request -
    """
    res = []

    formatted_name = f'"{name_param}":'

    name_tokens = [t.item() for t in small_llm.encode(formatted_name)[0]]
    res.extend(name_tokens)
    promt_tokenst.extend(name_tokens)
    term = "}" if is_last else ","
    try:
        vocab = get_vocab_list(small_llm)
    except OSError:
        return None
    term_id = vocab.get(term)
    bool_lst = ["TRUE", "FALSE", term]
    trie = Trie.to_trie(bool_lst, small_llm)
    node = trie.root

    while trie.get_name(node) is None:
        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        allowed_ids = trie.get_ids(node)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_ids] = 0
        masked_logits = logits + mask
        next_token_id = int(np.argmax(masked_logits))
        promt_tokenst.append(next_token_id)
        res.append(next_token_id)
        node = node.children[next_token_id]
    res.append(term_id)
    return res
