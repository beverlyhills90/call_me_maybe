import json
from functools import lru_cache
from typing import Any, cast

import numpy as np

from llm_sdk import Small_LLM_Model


@lru_cache(maxsize=None)
def get_vocab_list(small_llm: "Small_LLM_Model") -> dict[str, int]:
    """Get vocab list from vocab.json (loaded once per model instance)"""
    json_path = small_llm.get_path_to_vocab_file()
    try:
        with open(json_path, encoding="utf-8") as file:
            data = json.load(file)
        return cast(dict[str, int], data)
    except OSError as e:
        raise OSError("Something went wrong with the vocab.json") from e


def softmax(x: Any) -> Any:
    """softmax"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
