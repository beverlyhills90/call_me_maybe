import argparse
from pathlib import Path
from typing import Any


class CLIException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def cli_parsing_main() -> Any:
    """CLI arguments parsing

    raises CLIException
    """
    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description="Translate natural language prompts into function calls",
    )

    parser.add_argument(
        "--functions_definition",
        "-f",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        required=False,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        required=False,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/output/default_o.json"),
        required=False,
    )

    args = parser.parse_args()
    if len(vars(args)) > 3:
        raise CLIException("Need something more")
    elif len(vars(args)) < 3:
        raise CLIException("Too much args")

    for name, value in vars(args).items():
        if value is None:
            raise CLIException(f"Argument -{name} is missing!")

        str_value = str(value)
        if not str_value.endswith(".json"):
            raise CLIException(f"-{name}={value} is not a json file")
    return args
