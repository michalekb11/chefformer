. PHONY: help

help:
	@echo "Additional targets are pulled from the makefiles/ directory"

install-chefformer:
	pip install -e ".[test]"

freeze-requirements:
	pip freeze > requirements.lock

test-unit:
	pytest -q tests/unit

test-integration:
	pytest -q tests/integration

test-e2e:
	pytest -q tests/e2e

test-all:
	pytest -q tests

include makefiles/train.mk
include makefiles/inference.mk