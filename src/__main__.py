import os
import sys
from pydantic import Field, BaseModel
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_sdk import Small_LLM_Model
import src.json_parsing as jp
from src.arguments_generators_pack import number_generate,str_generator




ARGUMENT_PROMPT_TEMPLATE_NUM = """
Extract parameter "{arg_name}" from the user request.

Function: {function_name}
Description: {function_description}
Parameters: {parameters_description}

User request: "{user_request}"

The value of "{arg_name}" is: """

ARGUMENT_PROMPT_TEMPLATE_STR = """
You are a precise data extraction subroutine. Your task is to look at the "User Request", find the data that matches the "Description" of the requested parameter, and output ONLY that value.

CRITICAL RULES:
1. If the requested value is a punctuation mark or special character named as a word (e.g., "asterisks", "dashes", "dots", "slashes"), you MUST output the actual literal symbol (e.g., "*", "-", ".", "/"), NOT the word.
2. If the user request implies an empty value or empty string (e.g., '', ""), do not invent characters. Output NOTHING inside the quotes.
3. Output ONLY the raw extracted value inside double quotes. Do not repeat the parameter name, do not write "Value:", do not add explanations.

---
### EXAMPLES OF EXTRACTION ###

Requested Parameter: "recipient_email"
Description: The email address to send the message to.
User request: "Send an urgent email to boss@company.com saying I will be late"
[RESULT]: "boss@company.com"

Requested Parameter: "separator"
Description: The character used to split the text.
User request: "Split the string by dashes"
[RESULT]: "-"

Requested Parameter: "text_to_reverse"
Description: The original string that needs to be reversed.
User request: "Reverse the string ''"
[RESULT]: ""

Requested Parameter: "count"
Description: How many times to repeat.
User request: "Print 'Hello' 5 times"
[RESULT]: "5"

---
### CURRENT TASK ###
Requested Parameter: "{arg_name}"
Description: {parameters_description}
User request: "{user_request}"

[RESULT]: \""""

arguments_types_machine = {
    "number":number_generate,
    "string":str_generator,
}
arguments_types_promts = {
    "number":ARGUMENT_PROMPT_TEMPLATE_NUM,
    "string":ARGUMENT_PROMPT_TEMPLATE_STR,
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
def name_generator(prefix_trie:"Trie", small_llm:"Small_LLM_Model", promt_tokenst:list[int]) -> str:
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
def arguments_generator(small_llm:"Small_LLM_Model",arguments_list:list[tuple],function_desc,user_req:str) -> list[int]:
    result = []
    result.append(small_llm.encode("{")[0][0].item())

    lines = []
    for arg in arguments_list:
        lines.append(f"{arg[0]}:{arg[1]}")
    all_args_str = "\n".join(lines)

    for arg in arguments_list:
        arg_name, arg_type = arg
        promt_for_arg = arguments_types_promts[arg_type]
        generator_func = arguments_types_machine[arg_type]

        arguments_promt_str = promt_for_arg.format(arg_name=arg_name,function_name=function_desc[0],
            function_description=function_desc[1],parameters_description=all_args_str,user_request=user_req)
        promt_for_selector = small_llm.encode(arguments_promt_str)[0].tolist()
        is_last = (arg == arguments_list[-1])
        param_tokens = generator_func(small_llm,promt_for_selector,arg_name,is_last,user_req)
        result.extend(param_tokens)
    return result


def func_promt_generator(small_llm:"Small_LLM_Model",func_list:list[tuple],user_request:str):
    FUNCTION_CHOOSE_TEMPLATE = """

    You must choose exactly one function from the list below that best matches the user request.

    Available functions:
    {func_name_desc}

    User request: {user_request}

    Answer with only the function name.
    """

    lines = []
    for func in func_list:
        lines.append(f"{func[0]}:{func[1]}")
    all_functions_str = "\n".join(lines)
    full_prompt_str = FUNCTION_CHOOSE_TEMPLATE.format(func_name_desc=all_functions_str,user_request=user_request)
    tokens = small_llm.encode(full_prompt_str)[0].tolist()
    return tokens



def func_desc(function_name:str,func_list:list[tuple]) -> tuple | None:
    for func in func_list:
        if func[0] == function_name:
            return func
    return None

def main():
    small_llm = Small_LLM_Model()
    func_list = jp.parsing_name_desc()#all func names and desc
    all_funcs_names = [name[0] for name in func_list] #func names for trie
    prefix_trie = Trie.to_trie(all_funcs_names,small_llm)
    ur = "Reverse the string ' '" #user input
    function_name = small_llm.decode(name_generator(prefix_trie,small_llm,func_promt_generator(small_llm,func_list,ur)))
    print(function_name)
    parametrs_for_func = jp.arg_type(function_name)
    function_and_description = func_desc(function_name,func_list)
    args = arguments_generator(small_llm,parametrs_for_func,function_and_description,ur)
   
    print(small_llm.decode(args))
   

    

if __name__ == "__main__":
    main()
