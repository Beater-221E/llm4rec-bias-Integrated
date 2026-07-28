.PHONY: help install validate test smoke clean

help:
	@echo "Targets:"
	@echo "  install    pip install -r requirements.txt && pip install -e ."
	@echo "  validate   validate smoke_test config"
	@echo "  test       pytest tests/framework tests/unit"
	@echo "  smoke      validate"
	@echo "  clean      remove caches, runs, logs, processed data (keep data/raw)"

install:
	pip install -r requirements.txt
	pip install -e .

validate:
	PYTHONPATH=src python -m llm4rec.cli.main validate experiment=smoke_test

test:
	PYTHONPATH=src pytest -q tests/framework tests/unit

smoke: validate

clean:
	find . -type d -name '__pycache__' ! -path './.git/*' -print0 | xargs -0 rm -rf
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) ! -path './.git/*' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name '*.egg-info' ! -path './.git/*' -print0 | xargs -0 rm -rf
	@for d in runs logs experiments reports data/cache data/processed data/preprocessed; do \
		mkdir -p $$d; \
		find $$d -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} + 2>/dev/null || true; \
		touch $$d/.gitkeep; \
	done
	@mkdir -p logs/mllm4rec data/raw && touch logs/mllm4rec/.gitkeep data/raw/.gitkeep
	@echo "Cleaned caches and generated artifacts (data/raw kept)."
