import json
from pathlib import Path
from pydantic import Field, BaseModel, model_validator
import operator
import ast
import re


class Function(BaseModel):
    name: str = Field(min_length=5, max_length=100)
    description: str = Field(min_length=5, max_length=100)
    parameters: dict
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_parameters_structure(self) -> "Function":
        return self


def parsing_promts(file_path: str) -> list[str]:
    """Function to pras user promts

    ARGS:
    file_path - path to the file with user promts

    Raises: Exeption
    """
    current_dir = Path(__file__).resolve().parent.parent
    ret_list = []
    json_path = current_dir / file_path
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        raise e

    for promt in data:
        ret_list.append(promt.get("prompt"))
    return ret_list


def list_objects(file_path: str) -> list["Function"]:
    """Return list of objects "Function" after parsing
    ARG:
    file_path - path to the file with function definitions

    Raises: Exeption
    """
    current_dir = Path(__file__).resolve().parent.parent
    ret_list = []
    json_path = current_dir / file_path

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        raise e

    for func in data:
        ret_list.append(Function.model_validate(func))
    return ret_list


def safe_eval_math(value):
    """function for safe math operation in case if arguments like 8 - 4 was generated"""
    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, str):
        VALID_OPERATORS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        try:
            node = ast.parse(value, mode="eval").body

            if isinstance(node, ast.Constant):
                return node.value

            if isinstance(node, ast.BinOp) and type(node.op) in VALID_OPERATORS:
                left = safe_eval_math(
                    ast.unparse(node.left) if hasattr(ast, "unparse") else node.left
                )
                right = safe_eval_math(
                    ast.unparse(node.right) if hasattr(ast, "unparse") else node.right
                )
                return VALID_OPERATORS[type(node.op)](left, right)

        except Exception:
            return value

    return value


def write_to_file(
    file_path: str, function_name: str, promt: str, arguments: str
) -> None:
    """Write generated output to json output file
    Args:
    file_path - path to output file
    function_name - name of function
    promt - user request
    arguments - generated arguments for function

    """
    current_dir = Path(__file__).resolve().parent.parent
    user_path = Path(file_path)
    json_path = current_dir / user_path if not user_path.is_absolute() else user_path

    json_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arguments, str):
        arguments = re.sub(r"\\([dwisbDWISB])", r"\\\\\1", arguments)
    fixed_arguments = re.sub(
        r":\s*([0-9\s.+\-*/]+(?:[+\-*/][0-9\s.+\-*/]+)+)", r': "\1"', arguments
    )
    try:
        if isinstance(fixed_arguments, str):
            arguments_dict = json.loads(fixed_arguments)
        else:
            arguments_dict = fixed_arguments
    except json.JSONDecodeError:
        arguments_dict = {"raw_error_arguments": arguments}

    if isinstance(arguments_dict, dict):
        arguments_valid = {
            key: safe_eval_math(val) for key, val in arguments_dict.items()
        }
    else:
        arguments_valid = safe_eval_math(arguments_dict)

    arguments_valid = {key: safe_eval_math(val) for key, val in arguments_dict.items()}
    new_data = {"prompt": promt, "name": function_name, "parameters": arguments_valid}
    data = []

    if Path(json_path).exists():
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [data]
        except json.JSONDecodeError:
            data = []

    data.append(new_data)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
