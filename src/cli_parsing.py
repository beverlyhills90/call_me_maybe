
import argparse
from pathlib import Path
from typing import Any

class CliExeption(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def cli_parsing_main() -> Any:
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('-functions_definition')
    parser.add_argument('-input')
    parser.add_argument('-output')

    args = parser.parse_args()
    if len(vars(args)) > 3:
        raise  CliExeption("Need something more")
    elif len(vars(args)) < 3:
        raise  CliExeption("Too much args")
    for name,value in vars(args).items():
        if not value.endswith(".json"):
            raise CliExeption(f"{name}:{value} is not a json file")

        
    return args