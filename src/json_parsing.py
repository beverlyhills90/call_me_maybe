import json
from pathlib import Path

#parsing functions name for encode
def functions_parsing() -> list[str]:
    function_names = []
    current_dir = Path(__file__).resolve().parent
    
    json_path = current_dir / ".." / "data" / "input" / "functions_definition.json"
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for func in data:
        func_name = func.get('name')
        function_names.append(func_name)
    return function_names

#parsing funcs_name and description
def parsing_name_desc() -> list[tuple]:
    current_dir = Path(__file__).resolve().parent
    ret_list = []
    json_path = current_dir / ".." / "data" / "input" / "functions_definition.json"
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for func in data:
        func_name = func.get('name')
        func_desc = func.get('description')
        ret_list.append((func_name,func_desc))
    return ret_list



#parsing args for 
def arg_type(func_name:str) -> str:
    func_args = {}
    param_type = []
    current_dir = Path(__file__).resolve().parent
    
    json_path = current_dir / ".." / "data" / "input" / "functions_definition.json"
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for func in data:
        if func.get('name') == func_name:
            func_args = func['parameters']
    for item,value in func_args.items():
        upp = (item,str(list(value.values())[0]))
        param_type.append(upp)
    return param_type