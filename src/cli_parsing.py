import argparse
from pathlib import Path
from typing import Any


class CLIExeption(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def cli_parsing_main() -> Any:
    """CLI arguments parsing

    raises CLIExeption
    """
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "--functions_definition",
        "-f",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    if len(vars(args)) > 3:
        raise CLIExeption("Need something more")
    elif len(vars(args)) < 3:
        raise CLIExeption("Too much args")

    for name, value in vars(args).items():
        if value is None:
            raise CLIExeption(f"Argument -{name} is missing!")
            return

        str_value = str(value)
        if not str_value.endswith(".json"):
            raise CLIExeption(f"-{name}={value} is not a json file")
    return args
