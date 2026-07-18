import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class Parameter(BaseModel):
    type: str = Field(min_length=1, max_length=100)


class Function(BaseModel):
    name: str = Field(min_length=5, max_length=100)
    description: str = Field(min_length=5, max_length=100)
    parameters: dict[str, Parameter]
    returns: dict[str, str]


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)


def parsing_promts(file_path: str) -> list[str]:
    """Paras user promts

    ARGS:
    file_path - path to the file with user promts

    Raises: json.JSONDecodeError,FileNotFoundError
    """
    current_dir = Path(__file__).resolve().parent.parent
    ret_list = []
    json_path = current_dir / file_path
    try:
        with open(json_path, encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    for promt in data:
        try:
            validated = Prompt.model_validate(promt)
            ret_list.append(validated.prompt)
        except ValidationError:
            print(f"Validation Error in promt: {list(promt.values())[0]}")
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
        with open(json_path, encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    for func in data:
        ret_list.append(Function.model_validate(func))
    return ret_list


def write_to_file(
    file_path: str,
    function_name: str,
    prompt: str,
    arguments: str,
    args_schema: Any,
) -> None:
    """Write generated output to json output file
    Args:
    file_path - path to output file
    function_name - name of function
    prompt - user request
    arguments - generated arguments for function

    """
    current_dir = Path(__file__).resolve().parent.parent
    user_path = Path(file_path)
    if user_path.is_absolute():
        json_path = user_path
    else:
        json_path = current_dir / user_path

    json_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(arguments, str):
        arguments = re.sub(r"\\([dDwWsS])", r"\\\\\1", arguments)

    fixed_arguments = re.sub(
        r'(?<!\\)\\(?!\\|"|\\/|n|r|t|b|f|u)', r"\\\\", arguments
    )

    try:
        if isinstance(fixed_arguments, str):
            arguments_dict = json.loads(fixed_arguments)
        else:
            arguments_dict = fixed_arguments
    except json.JSONDecodeError:
        arguments_dict = {"raw_error_arguments": arguments}

    if isinstance(arguments_dict, dict) and args_schema is not None:
        for key in list(arguments_dict.keys()):
            if key in args_schema:
                param_type = args_schema[key].type
                if param_type == "number":
                    try:
                        arguments_dict[key] = float(arguments_dict[key])
                    except (ValueError, TypeError):
                        pass
                elif param_type == "string":
                    try:
                        arguments_dict[key] = (
                            str(arguments_dict[key]).strip().rstrip("'")
                        )
                    except (ValueError, TypeError):
                        pass

    new_data = {
        "prompt": prompt,
        "name": function_name,
        "parameters": arguments_dict,
    }

    data = []

    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as file:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [data]
        except json.JSONDecodeError:
            data = []

    data.append(new_data)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
