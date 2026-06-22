import os
import sys
from pydantic import Field, BaseModel
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_sdk import Small_LLM_Model
import json_parsing as jp
from arguments_generators_pack import number_generate,str_generator
import dotenv

dotenv.load_dotenv()

arg_prompt = f"""

    You must choose exactly one function from the list below that best matches the user request.

    Available functions:
    - fn_add_numbers: Add two numbers together and return their sum.
    - fn_greet: Generate a greeting message for a person by name.
    - fn_reverse_string: Reverse a string and return the reversed result.
    - fn_get_square_root:Calculate the square root of a number
    - fn_substitute_string_with_regex: Replace all occurrences matching a regex pattern in a string.

    User request:  "Reverse the string 'hello'"

    Answer with only the function name.
    """



arguments_promt_str = f"""
    You are a function calling assistant. Generate the arguments for the 
    function below based on the user request.

    Function: fn_reverse_string
    Description: Reverse a string and return the reversed result.
    Parameters:
    - s (string)

    User request: "Reverse the string 'hello'"

    Generate the JSON arguments:
    """

ARGUMENT_PROMPT_TEMPLATE_NUM = """
Extract parameter "{name_param}" from the user request.

Example: For request "What is the sum of 10 and 5?":
- a = 10
- b = 5

User request: "{user_request}"

The value of "{name_param}" is: """

ARGUMENT_PROMPT_TEMPLATE_STR= """
Extract parameter "{arg_name}" from the user request.

Example: For request "Replace all 'cat' with 'dog' in 'cat is here'":
- source_string = "cat is here"
- regex = "cat"  
- replacement = "dog"

User request: "Replace all 'hello' with 'world' in 'hello world'"

The value of "{arg_name}" is: \"
"""

all_funcs = ["fn_add_numbers","fn_greet","fn_reverse_string","fn_get_square_root","fn_substitute_string_with_regex"]

finite_state_machine = {
    "number":number_generate,
    "string":str_generator,
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


#function Name Generator
def name_selector(prefix_trie:"Trie", small_llm:"Small_LLM_Model", promt_tokenst:list[int]) -> str:
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


#wraper for arguments
def arguments_generator(small_llm:"Small_LLM_Model",arguments_list:list[tuple]) -> list[int]:
    result = []
    result.append(small_llm.encode("{")[0][0].item())
    for arg in arguments_list:
        arg_name, param_type = arg
        generator_func = finite_state_machine[param_type]
        arguments_promt_str = ARGUMENT_PROMPT_TEMPLATE_STR.format(arg_name=arg_name)
        promt_for_args = small_llm.encode(arguments_promt_str)[0].tolist()
        is_last = (arg == arguments_list[-1])
        param_tokens = generator_func(small_llm,promt_for_args,arg_name,is_last)
        result.extend(param_tokens)
    return result



def main():
    small_llm = Small_LLM_Model()
    prefix_trie = Trie.to_trie(all_funcs,small_llm)
    promt_tokens = small_llm.encode(arg_prompt)[0].tolist()
    function_name = "fn_substitute_string_with_regex" #small_llm.decode(name_selector(prefix_trie,small_llm,promt_tokens))
    parametrs_for_func = jp.parametr_type(function_name)
    res = arguments_generator(small_llm,parametrs_for_func)
   
    print(small_llm.decode(res))
   

    

if __name__ == "__main__":
    main()
