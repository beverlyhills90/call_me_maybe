PYTHON = python3
UV = uv

.PHONY: all install run debug clean lint

all: install lint run

install:
	$(UV) sync

run:
	$(UV) run python -m src --functions_definition data/input/functions_definition.json --input data/input/test_1.json --output data/output/default_o.json

debug:
	$(PYTHON) -m pdb src/__main__.py

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +

lint:
	uv run mypy . --exclude .venv --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	uv run flake8 . --exclude=.venv
	