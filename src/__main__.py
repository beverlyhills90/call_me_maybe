import os
import sys
from pydantic import Field, BaseModel, model_validator
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_sdk import Small_LLM_Model
import src.json_part as jp
from src.arguments_generators_pack import number_generate, str_generator, bool_generate
from src.cli_parsing import cli_parsing_main, CLIExeption
import src.promts as promts
from src.promts import Node,Trie
from typing import cast
from json import JSONDecodeError

arguments_types_machine = {
    "number": number_generate,
    "string": str_generator,
    "boolean": bool_generate,
}
arguments_types_promts = {
    "number": promts.ARGUMENT_PROMPT_TEMPLATE_NUM,
    "string": promts.ARGUMENT_PROMPT_TEMPLATE_STR,
    "boolean": promts.ARGUMENT_PROMPT_TEMPLATE_BOOL,
}




# function Name Generator
def name_generator(
    prefix_trie: "Trie", small_llm: "Small_LLM_Model", promt_tokenst: list[int]
) -> list[int]:
    node = prefix_trie.root
    res = []

    while prefix_trie.get_name(node) is None:
        logits = small_llm.get_logits_from_input_ids(promt_tokenst)
        allowed_ids = prefix_trie.get_ids(node)
        mask = np.full(len(logits), -np.inf)
        mask[allowed_ids] = 0
        masked_logits = logits + mask
        next_token_id = int(np.argmax(masked_logits))
        promt_tokenst.append(next_token_id)
        res.append(next_token_id)
        node = node.children[next_token_id]
    return res


def arguments_generator(
    small_llm: "Small_LLM_Model",
    arguments_list: list[tuple],
    function_desc:tuple[str, str],
    user_req: str,
) -> list[int]:
    """wraper for arguments generators"""
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
            promt_for_arg = promts.ARGUMENT_PROMPT_TEMPLATE_STR_REGEX
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
        param_tokens = generator_func(small_llm, promt_for_selector, arg_name, is_last)
        result.extend(param_tokens)
    return result


def func_promt_generator(
    small_llm: "Small_LLM_Model", func_list: list[tuple], user_request: str
) -> list[int]:
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
    return cast(list[int], tokens)


def from_dict_to_list(target_dict: dict) -> list[tuple]:
    """Converts a dictionary of parameters into a list of tuples of the form (name, type)."""
    res = []
    for item, value in target_dict.items():
        param_type = vars(value).get("type")
        res.append((item, param_type))
    return res


def main() -> None:
    """main()"""
    try:
        cli = cli_parsing_main()
    except CLIExeption as e:
        print(f"Somtehing wrong with arguments: {e}")
        return
    small_llm = Small_LLM_Model()
    try:
        func_list = jp.list_objects(cli.functions_definition)
    except Exception as e:
        print(f"\033[31m{e}\033[0m")
        return
    all_funcs_names = [obj.name for obj in func_list]
    func_descriptions = [obj.description for obj in func_list]
    func_tuples = list(zip(all_funcs_names, func_descriptions))
    prefix_trie = Trie.to_trie(all_funcs_names, small_llm)
    try:
        user_input = jp.parsing_promts(cli.input)
    except (JSONDecodeError, FileNotFoundError) as e:
        print(e)
        return
    
    print("T-3000 working on your promts")
    i:int = 0
    for request in user_input:
        crasota = f"[{"=" * ( i * 10)}{" " * ((len(user_input) - i) * 10)}]"
        print(crasota,end="\r",flush=True)
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
        
        i += 1


if __name__ == "__main__":
    main()
