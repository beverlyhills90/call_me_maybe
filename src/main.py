import os
import sys
from pydantic import Field, BaseModel
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_sdk import Small_LLM_Model
from enum import Enum

arg_prompt = f"""

    You must choose exactly one function from the list below that best matches the user request.

    Available functions:
    - fn_add_numbers: Add two numbers together and return their sum.
    - fn_greet: Generate a greeting message for a person by name.
    - fn_reverse_string: Reverse a string and return the reversed result.
    - fn_get_square_root:Calculate the square root of a number
    - fn_substitute_string_with_regex: Replace all occurrences matching a regex pattern in a string.

    User request:  "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"

    Answer with only the function name.
    """


class State(Enum):
    

arguments_promt = f"""

    """

all_funcs = ["fn_add_numbers","fn_greet","fn_reverse_string","fn_get_square_root","fn_substitute_string_with_regex"]

finite_state_machine = {

}

class Node(BaseModel):
    children: dict[int, "Node"] = Field(default_factory=dict)
    function_name: str | None = None


class Trie(BaseModel):
    root:Node = Node()

    def insert(self,ids:list[int],function_name:str):
        node = self.root
        for id in ids:
            if id not in node.children:
                node.children[id] = Node()
            node = node.children[id]
        node.function_name = function_name

    @staticmethod
    def get_ids(node:"Node") -> list[int]:
        lst = []
        return node.children.keys()
    
    @staticmethod
    def get_name(node:"Node") -> str | None:
        return node.function_name

    @classmethod
    def to_trie(cls,all_funcs:list[str],small_llm:"Small_LLM_Model") -> "Trie":
        trie = cls()
        for func in all_funcs:
            tmp = small_llm.encode(func)[0].tolist()
            trie.insert(tmp,func)
        return trie



def cicle(prefix_trie:"Trie", small_llm:"Small_LLM_Model", promt_tokenst:list[int]) -> str:
    node = prefix_trie.root
    res = []
    while prefix_trie.get_name(node) is None:
        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        allowed_ids = list(prefix_trie.get_ids(node))
        mask = np.full(len(logits), -np.inf)
        mask[allowed_ids] = 0
        masked_logits = logits + mask
        next_token_id = int(np.argmax(masked_logits))
        promt_tokenst.append(next_token_id)
        res.append(next_token_id)
        node = node.children[next_token_id]
    return res

def main():
    small_llm = Small_LLM_Model()
    prefix_trie = Trie.to_trie(all_funcs,small_llm)
    promt_tokens = small_llm.encode(arg_prompt)[0].tolist()
    function_name = small_llm.decode(cicle(prefix_trie,small_llm,promt_tokens))
    print(function_name)
   

    

if __name__ == "__main__":
    main()
