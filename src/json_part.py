import json
from pathlib import Path
from pydantic import Field, BaseModel,model_validator


class Function(BaseModel):
    name:str = Field(min_length=5,max_length=100)
    description:str = Field(min_length=5,max_length=100)
    parameters:dict 
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_parameters_structure(self) -> "Function":
        return self



def parsing_promts(file_name:str | None = None) -> list[str]:
    current_dir = Path(__file__).resolve().parent
    ret_list = []
    json_path = current_dir / ".." / "data" / "input" / "function_calling_tests.json"
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for promt in data:
        ret_list.append(promt.get('prompt'))
    return ret_list


def list_objects(file_name:str | None = None) -> list["Function"]:
    current_dir = Path(__file__).resolve().parent
    ret_list = []
    json_path = current_dir / ".." / "data" / "input" / "functions_definition.json"
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for func in data:
        ret_list.append(Function.model_validate(func))
    return ret_list
