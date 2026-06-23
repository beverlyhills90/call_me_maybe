import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_sdk import Small_LLM_Model
from enum import Enum
import json
from typing import Any
from pydantic import BaseModel, Field


class Node(BaseModel):
    children: dict[int, "Node"] = Field(default_factory=dict)
    function_name: str | None = None


class Trie(BaseModel):
    root: Node = Node()

    def insert(self, ids: list[int], function_name: str):
        node = self.root
        for id in ids:
            if id not in node.children:
                node.children[id] = Node()
            node = node.children[id]
        node.function_name = function_name

    @staticmethod
    def get_ids(node: "Node") -> list[int]:
        lst = []
        return node.children.keys()

    @staticmethod
    def get_name(node: "Node") -> str | None:
        return node.function_name

    @classmethod
    def to_trie(cls, all_funcs: list[str], small_llm: "Small_LLM_Model") -> "Trie":
        trie = cls()
        for func in all_funcs:
            tmp = small_llm.encode(func)[0].tolist()
            trie.insert(tmp, func)
        return trie


def get_vocab_list(small_llm: "Small_LLM_Model"):

    json_path = small_llm.get_path_to_vocab_file()
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    res = []
    return data


def softmax(x: Any, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def bool_generate(
    small_llm: "Small_LLM_Model",
    promt_tokenst: list[int],
    name_param: str,
    is_last: bool,
    user_request: str,
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
    bool_lst = ["TRUE", "FALSE", term]
    trie = Trie.to_trie(bool_lst, small_llm)
    node = trie.root

    while trie.get_name(node) is None:
        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        allowed_ids = list(trie.get_ids(node))
        mask = np.full(len(logits), -np.inf)
        mask[allowed_ids] = 0
        masked_logits = logits + mask
        next_token_id = int(np.argmax(masked_logits))
        promt_tokenst.append(next_token_id)
        res.append(next_token_id)
        node = node.children[next_token_id]
    res.append(term_id)
    return res
