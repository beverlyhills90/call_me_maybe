from pydantic import Field, BaseModel, model_validator
from llm_sdk import Small_LLM_Model

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


class Node(BaseModel):
    children: dict[int, "Node"] = Field(default_factory=dict)
    function_name: str | None = None


class Trie(BaseModel):
    root: Node = Node()

    def insert(self, ids: list[int], function_name: str) -> None:
        node = self.root
        for id in ids:
            if id not in node.children:
                node.children[id] = Node()
            node = node.children[id]
        node.function_name = function_name

    @staticmethod
    def get_ids(node: "Node") -> list[int]:
        return list(node.children.keys())

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
