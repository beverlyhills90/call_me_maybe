*This project has been created as part of the 42 curriculum by oldanyli.*

---

## Description

**call me maybe** is a function calling system that translates natural language prompts into structured JSON function calls using a small 0.6B parameter language model (Qwen3-0.6B). Given a user request like *"What is the sum of 2 and 3?"*, the system outputs not an answer, but a structured function call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2, "b": 3}
}
```

The core challenge is reliability — small models produce valid JSON only ~30% of the time when prompted naively. This project implements **constrained decoding** from scratch, achieving 100% structurally valid JSON output by guiding token selection at every generation step.

---

## Instructions

**Requirements:** Python 3.10+, uv

**Installation:**
```bash
make install
```

**Run:**
```bash
make run
```

Or with custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

**Lint:**
```bash
make lint
```

> By default, the program reads from `data/input/` and writes to `data/output/`. Custom paths can be specified via `--functions_definition`, `--input`, and `--output` arguments.

---

## Pipeline Overview

```
Input Files
    │
    ├── functions_definition.json ──► Parse & Validate (pydantic)
    │                                        │
    └── function_calling_tests.json ─────────┤
                                             │
                                             ▼
                                    For each prompt:
                                             │
                                    ┌────────▼────────┐
                                    │   Stage 1:      │
                                    │   Function      │
                                    │   Selection     │
                                    │  (Prefix Trie)  │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   Stage 2:      │
                                    │   Argument      │
                                    │   Generation    │
                                    │  (FSM + vocab)  │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    Assemble result:
                               {prompt, name, parameters}
                                             │
                                             ▼
                                function_calls.json
```

---

## Algorithm Explanation

### Stage 1 — Function Selection via Prefix Trie

Function names from `functions_definition.json` are tokenized and inserted into a **prefix trie** (one node per token ID). During generation, at each step only tokens that are children of the current trie node are unmasked — all others are set to `-inf`. The model navigates the trie token by token until reaching a leaf node, which identifies the selected function. This guarantees the output is always a valid, existing function name.

```
Prompt → encode() → token IDs → generation loop:

                        root
                          │
                    [token: fn_]
                          │
              ┌───────────┼───────────┐
           [add]        [greet]    [reverse]  ...
              │            │           │
         [numbers]       (leaf)    [string]
              │             │          │
           (leaf)      fn_greet    (leaf)
              │                        │
       fn_add_numbers          fn_reverse_string

At each step:
  logits = LLM(current_sequence)
  allowed = children of current node
  mask everything else → -inf
  next_token = argmax(logits)
  move to child node
  if leaf → function name found
```

### Stage 2 — Argument Generation via Finite State Machine

For each parameter, a dedicated prompt is constructed including the function name, description, and the specific parameter being extracted. Token masks are built by filtering the model's vocabulary (`vocab.json`) to include only tokens valid for the current type and position.

A finite state machine tracks the current generation state and adjusts the allowed token set at every step.

```
For each parameter:
  1. Insert "param_name": directly (no LLM needed)
  2. Run FSM to generate value

NUMBER FSM:

  START_NUMS ──────────────────► JUST_NUMBERS
  allowed:                       allowed:
  [0-9, -]                       [0-9, term]
                                      │
                               term selected
                                      │
                                 END_NUMS ✓

STRING FSM:

  (opening " inserted directly)
        │
        ▼
  JUST_SYMBOLS ──────────────► AFTER_STR
  allowed:                     allowed:
  [all vocab                   [term]
   tokens except "]               │
   + " as closing signal]         ▼
        │                    END_STR ✓
   " selected
        │
        ▼
   AFTER_STR

Token masks built from vocab.json:
  number → tokens where all chars in "0123456789"
  string → tokens where '"' not in token
```

---

## Design Decisions

**Separate prompt per parameter** — rather than asking the model to generate all arguments at once, a dedicated prompt is constructed for each parameter individually. This significantly improves extraction accuracy, especially for functions with multiple parameters of similar types.

**Vocabulary filtering for token masks** — instead of hardcoding a list of character IDs, allowed tokens are derived dynamically by filtering `vocab.json`. This makes the system work correctly regardless of how the tokenizer encodes specific characters, and generalizes to any vocabulary.

**Specialized prompt for `fn_substitute_string_with_regex`** — this function has three string parameters with semantically distinct roles. A few-shot example prompt was added specifically for this function to help the model distinguish between `source_string`, `regex`, and `replacement`.

**Prefix trie for function name generation** — instead of relying on the model to spell the function name correctly, a trie built from tokenized function names guarantees that only valid, existing function names can be generated. Typos and hallucinated names are structurally impossible.

---

## Performance Analysis

- Tested on the standard test file — **100% accuracy** on expected outputs
- Stress-tested with non-standard inputs (numbers as words, ambiguous prompts, special characters) — accuracy remains **90%+**
- All outputs are **100% structurally valid JSON** — guaranteed by constrained decoding regardless of model behavior

---

## Challenges Faced

Understanding and implementing constrained decoding from scratch required experimenting with several approaches to the finite state machine design. Each state transition and token mask had to be carefully tuned — for example, discovering that vocabulary filtering was necessary because `encode('.')` returns a different token ID than what the model produces in a numeric context. Each component (trie, number FSM, string FSM) was built and tested independently before integration.

---

## Testing Strategy

Validated against the provided `function_calling_tests.json` examples. Additionally, stress-tested with edge cases including:

- Numbers written as words (`"two"`, `"three"`)
- Strings with special characters and apostrophes
- Multi-parameter functions with semantically similar parameters (`fn_substitute_string_with_regex`)
- Ambiguous prompts

---

## Example Usage

```bash
# Default paths
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

**Input** (`function_calling_tests.json`):
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet john"}
]
```

**Output** (`function_calls.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
  },
  {
    "prompt": "Greet john",
    "name": "fn_greet",
    "parameters": {"name": "john"}
  }
]
```

---

## Bonus Features

### Custom BPE Tokenizer

A public implementation of `encode()` and `decode()` methods was written from scratch, avoiding direct use of the SDK's built-in tokenizer methods in the main code. Only `get_path_to_vocab_file()` and `get_path_to_merges_file()` (public SDK methods) are used internally.

```
Custom BPE Tokenizer — encode() pipeline:

  Input text: "hello world"
       │
       ▼
  UTF-8 bytes → Unicode mapping:
  space → Ġ,  \n → Ċ,  \r → ĉ,  other → chr(byte)
       │
       ▼
  Split into individual chars:
  ['h', 'e', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']
       │
       ▼
  BPE merge loop (merges.txt):

  Iteration 1: find highest priority pair from merges.txt
  ['h', 'e', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']
   best pair: ('Ġ', 'w') → rank 42
   ['h', 'e', 'l', 'l', 'o', 'Ġw', 'o', 'r', 'l', 'd']

  Iteration 2: next highest priority pair
   best pair: ('l', 'o') → rank 87
   ['h', 'e', 'l', 'lo', 'Ġw', 'o', 'r', 'l', 'd']

  ... repeat until no more pairs in merges.txt ...

  Final tokens: ['hello', 'Ġworld']
       │
       ▼
  Lookup in vocab.json:
  'hello' → 9707
  'Ġworld' → 1879
       │
       ▼
  Output: [9707, 1879]

decode() — reverse pipeline:

  [9707, 1879]
       │
       ▼
  Invert vocab.json:
  9707 → 'hello'
  1879 → 'Ġworld'
       │
       ▼
  Unicode → UTF-8:
  'Ġ' → ' ',  'Ċ' → '\n',  'ĉ' → '\r'
       │
       ▼
  Output: "hello world"
```

---

## Resources

- **3Blue1Brown** — Visual series on how LLMs work: https://www.youtube.com/watch?v=wjZofJX0v4M
- **Andrej Karpathy** — Let's build the GPT Tokenizer: https://youtu.be/zduSFxRajkE
- **A Guide to Structured Generation Using Constrained Decoding**: https://www.aidancooper.co.uk/constrained-decoding/

**AI usage:** Claude was used throughout this project for conceptual guidance — explaining constrained decoding, trie structures, finite state machines, and tokenization theory. All implementation decisions and code were written and understood by the author independently.