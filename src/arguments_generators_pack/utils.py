import json
from typing import Any, cast

import numpy as np

from llm_sdk import Small_LLM_Model


def get_vocab_list(small_llm: "Small_LLM_Model") -> dict[str, int]:
    """Get vocab list from vocab.json"""
    json_path = small_llm.get_path_to_vocab_file()
    try:
        with open(json_path, encoding="utf-8") as file:
            data = json.load(file)
        return cast(dict[str, int], data)
    except OSError:
        raise OSError("Something went worng with the vocab.json")
        print("Something went worng with the vocab.json")


def softmax(x: Any) -> Any:
    """softmax"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
