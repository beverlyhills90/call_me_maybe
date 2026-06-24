from llm_sdk import Small_LLM_Model
import json
from typing import Any
import numpy as np
from typing import cast


def get_vocab_list(small_llm: "Small_LLM_Model") -> dict[str, int]:
    """Get vocab list from vocab.json"""
    json_path = small_llm.get_path_to_vocab_file()
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return cast(dict[str, int], data)


def softmax(x: Any) -> Any:
    """softmax"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
