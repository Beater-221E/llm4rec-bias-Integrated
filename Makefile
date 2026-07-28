.PHONY: help install validate test smoke

help:
	@echo "Targets:"
	@echo "  install    pip install -r requirements.txt && pip install -e ."
	@echo "  validate   validate smoke_test config"
	@echo "  test       pytest tests/unit"
	@echo "  smoke      validate"

install:
	pip install -r requirements.txt
	pip install -e .

validate:
	python -m llm4rec_bias_Integrated.cli.main validate experiment=smoke_test

test:
	pytest -q tests/unit

smoke: validate
