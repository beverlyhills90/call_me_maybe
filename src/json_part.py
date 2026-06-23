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



def parsing_promts(file_path:str) -> list[str]:
    current_dir = Path(__file__).resolve().parent.parent
    ret_list = []
    json_path = current_dir /file_path
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        raise e

    for promt in data:
        ret_list.append(promt.get('prompt'))
    return ret_list


def list_objects(file_path:str) -> list["Function"]:
    current_dir = Path(__file__).resolve().parent.parent
    ret_list = []
    json_path = current_dir /file_path
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        raise e

    for func in data:
        ret_list.append(Function.model_validate(func))
    return ret_list


def write_to_file(file_path:str,function_name:str,promt:str,arguments:str):
    current_dir = Path(__file__).resolve().parent.parent
    ret_list = []
    user_path = Path(file_path)
    json_path = current_dir / user_path if not user_path.is_absolute() else user_path

    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    new_data = {"prompt":promt,
            "name":function_name,
            "parameters":arguments}
    data = []
    
    if Path(json_path).exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    data = [data]
        except json.JSONDecodeError:
            data = []

    data.append(new_data)
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data,file,ensure_ascii=False, indent=4)


    