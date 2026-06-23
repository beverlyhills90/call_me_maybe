import os
import sys
from pydantic import Field, BaseModel, model_validator
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_sdk import Small_LLM_Model
import src.json_part as jp
from src.arguments_generators_pack import number_generate, str_generator, bool_generate
from src.cli_parsing import cli_parsing_main, CLIExeption

ARGUMENT_PROMPT_TEMPLATE_NUM = """You are a precise data extraction subsystem. Your task is to extract the EXACT value for the parameter "{arg_name}" from the user request and format it strictly as a JSON field.

Context:
- Function to call: {function_name}
- Function Description: {function_description}
- All Parameters Meta: {parameters_description}

Rules:
1. Extract ONLY the value for "{arg_name}".
2. Do not explain anything. Do not add spaces after the number.
3. End your response immediately with a closing curly brace '}}' if this is the end of the JSON.

Few-Shot Examples:
User request: "Set the temperature to 22 degrees"
The value of "temperature" is: 22}}

User request: "What is the sum of 'three' and five?"
fn_add_numbers
The value of "a" is: 3
User request: "The level should be 5"
The value of "level" is: 5}}

User request: "The level should be 7"
The value of "level" is: 7}}

User request: "What is the sum of and"
The value of "a" is: null


Current Task:
User request: "{user_request}"
The value of "{arg_name}" is: """

ARGUMENT_PROMPT_TEMPLATE_STR = """
You are a precise data extraction subroutine.

Extract the value of parameter "{arg_name}" from the user request.

Function: {function_name}
Description: {function_description}

Example:
Request: "Reverse the string 'hello'"
- s = "hello"
User request: "Reverse the string. Actually don't never mind."
The value of "s" is: ""

User request: "Undo the last text flipping"
The value of "s" is: ""

User request: "{user_request}"

The value of "{arg_name}" is: \"
"""

ARGUMENT_PROMPT_TEMPLATE_STR_REGEX = """You are a precise NLP backend routine. Convert the user request into a strict YAML block containing the exact regex arguments.

CRITICAL REGEX DICTIONARY:
- "vowels" -> '[aeiouAEIOU]'
- "numbers" / "digits" -> '\\d+'
- "words" -> '[a-zA-Z]+'
- specific words (like 'cat') -> 'cat'

Format your response EXACTLY like the examples below. Do not add any text before or after the YAML block.

Few-Shot Examples:
User request: "Replace all numbers in 'Hello 34' with NUMBERS"
---
source_string: "Hello 34"
regex: "\\d+"
replacement: "NUMBERS"
---

User request: "Extract all digits from 'Room 404 and 200'"
---
source_string: "Room 404 and 200"
regex: "\\d+"
replacement: ""
---

User request: "Replace all vowels in 'Programming is fun' with asterisks"
---
source_string: "Programming is fun"
regex: "[aeiouAEIOU]"
replacement: "*"
---

User request: "Substitute the word 'cat' with 'dog' in 'The cat sat'"
---
source_string: "The cat sat"
regex: "cat"
replacement: "dog"
---

User request: "Extract all digits from 'Room 404 and 200'"
---
source_string: "Room 404 and 200"
regex: "\\d+"
replacement: ""
---

Current Task:
User request: "{user_request}"
---
"""


ARGUMENT_PROMPT_TEMPLATE_BOOL = """Context:
- Function to call: {function_name}
- Function Description: {function_description}
- Parameters Metadata: {parameters_description}

Few-Shot Examples:

User request: "Turn on the dark mode feature"
fn_toggle_feature
{{"feature_name": "dark_mode", "enable": true}}

User request: "Deactivate notifications please"
fn_toggle_feature
{{"feature_name": "notifications", "enable": false}}

User request: "Greet Shrek"
fn_greet
{{"name": "Shrek"}}

Current Task:
User request: "{user_request}"
"""

arguments_types_machine = {
    "number": number_generate,
    "string": str_generator,
    "boolean": bool_generate,
}
arguments_types_promts = {
    "number": ARGUMENT_PROMPT_TEMPLATE_NUM,
    "string": ARGUMENT_PROMPT_TEMPLATE_STR,
    "boolean": ARGUMENT_PROMPT_TEMPLATE_BOOL,
}


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


# function Name Generator
def name_generator(
    prefix_trie: "Trie", small_llm: "Small_LLM_Model", promt_tokenst: list[int]
) -> str:
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


# wraper for arguments
def arguments_generator(
    small_llm: "Small_LLM_Model",
    arguments_list: list[tuple],
    function_desc,
    user_req: str,
) -> list[int]:
    result = []
    result.append(small_llm.encode("{")[0][0].item())

    lines = []
    for arg in arguments_list:
        lines.append(f"{arg[0]}:{arg[1]}")
    all_args_str = "\n".join(lines)

    for arg in arguments_list:
        arg_name, arg_type = arg

        generator_func = arguments_types_machine[arg_type]
        if "regex" in function_desc[0]:
            promt_for_arg = ARGUMENT_PROMPT_TEMPLATE_STR_REGEX
        else:
            try:
                promt_for_arg = arguments_types_promts[arg_type]
            except KeyError as e:
                print(f"{e}:We don't support this data type")
                continue
        arguments_promt_str = promt_for_arg.format(
            arg_name=arg_name,
            function_name=function_desc[0],
            function_description=function_desc[1],
            parameters_description=all_args_str,
            user_request=user_req,
        )
        promt_for_selector = small_llm.encode(arguments_promt_str)[0].tolist()
        is_last = arg == arguments_list[-1]
        param_tokens = generator_func(
            small_llm, promt_for_selector, arg_name, is_last
        )
        result.extend(param_tokens)
    return result


def func_promt_generator(
    small_llm: "Small_LLM_Model", func_list: list[tuple], user_request: str
):
    FUNCTION_CHOOSE_TEMPLATE = """

    You must choose exactly one function from the list below that best matches the user request.
    
    Example:
    User request: "Check if the firewall is there"
    Explanation: The user is asking about a system feature/component status (firewall), which belongs to feature management, not math.
    Selected Function: fn_toggle_feature

    User request: "Extract all digits from 'Room 404 and 200'"
    Explanation: The user wants to find text patterns (digits) inside a string using modern NLP/Regex, not perform a mathematical square root operation.
    Selected Function: fn_substitute_string_with_regex
    Available functions:
    {func_name_desc}

    User request: {user_request}

    Answer with only the function name.
    """

    lines = []
    for func in func_list:
        lines.append(f"{func[0]}:{func[1]}")
    all_functions_str = "\n".join(lines)
    full_prompt_str = FUNCTION_CHOOSE_TEMPLATE.format(
        func_name_desc=all_functions_str, user_request=user_request
    )
    tokens = small_llm.encode(full_prompt_str)[0].tolist()
    return tokens


def from_dict_to_list(target_dict: dict):
    res = []
    for item, value in target_dict.items():
        param_type = value.get("type")
        res.append((item, param_type))
    return res


def main() -> None:
    try:
        cli = cli_parsing_main()
    except CLIExeption as e:
        print(f"Somtehing wrong with arguments: {e}")
        return
    small_llm = Small_LLM_Model()
    try:
        func_list = jp.list_objects(cli.functions_definition)
    except Exception as e:
        print(e)
        return
    all_funcs_names = [obj.name for obj in func_list]
    func_descriptions = [obj.description for obj in func_list]
    func_tuples = list(zip(all_funcs_names, func_descriptions))
    prefix_trie = Trie.to_trie(all_funcs_names, small_llm)
    try:
        user_input = jp.parsing_promts(cli.input)
    except Exception as e:
        print(e)
        return

    for request in user_input:
        target_name = small_llm.decode(
            name_generator(
                prefix_trie,
                small_llm,
                func_promt_generator(small_llm, func_tuples, request),
            )
        )
        for func_obj in func_list:
            if func_obj.name == target_name:
                found_parameters = from_dict_to_list(func_obj.parameters)
                break
        function_and_description = (
            target_name,
            func_descriptions[all_funcs_names.index(target_name)],
        )
        args = arguments_generator(
            small_llm, found_parameters, function_and_description, request
        )
        decoded = small_llm.decode(args)
        jp.write_to_file(cli.output, target_name, request, decoded)


if __name__ == "__main__":
    main()
