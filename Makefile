# Audio Ops Framework: common commands. `make help` lists them.
# Targets use the .venv created by `make setup` when it exists, so you don't
# have to activate it first.
.PHONY: help setup setup-dev install demo run dev serve test test-fast lint format typecheck \
        docs docs-serve build up up-inference up-all down logs ps images new-task clean

CORE    := docker compose
INFER   := docker compose -f docker-compose.inference.yml
ALL     := docker compose -f docker-compose.yml -f docker-compose.inference.yml
BIN     := $(if $(wildcard .venv/bin/python),.venv/bin/,)
PY      ?= $(BIN)python
TASK    ?= example_classifier

help:
	@echo "Audio Ops Framework"
	@echo "  make setup         One-time setup: .venv, PyTorch, the framework, .env, demo audio"
	@echo "  make setup-dev     The same, plus test, lint and docs tools (contributors)"
	@echo "  make run           Train, evaluate and score TASK (default: example_classifier)"
	@echo "  make demo          Regenerate the synthetic demo audio"
	@echo "  make dev           Run Dagster locally (http://localhost:3000)"
	@echo "  make serve         Run the inference API locally (http://localhost:8000)"
	@echo "  make test          Full test suite (incl. end-to-end pipeline + API)"
	@echo "  make test-fast     Unit tests only"
	@echo "  make lint|format   Ruff lint / autofix"
	@echo "  make docs          Build the documentation site (Zensical) into ./site"
	@echo "  make docs-serve    Live-preview the docs"
	@echo "  make build         Build sdist + wheel into ./dist"
	@echo "  make up            Docker: core (Dagster)"
	@echo "  make up-inference  Docker: streaming inference + MediaMTX"
	@echo "  make up-all        Docker: both"
	@echo "  make down | logs | ps | images"
	@echo "  make new-task NAME=my_task [TYPE=regression]"
	@echo "  make clean         Remove local runtime artifacts"

setup:
	bash scripts/setup.sh

setup-dev:
	bash scripts/setup.sh --dev

install: setup-dev

demo:
	@mkdir -p mediamtx/test_audio
	$(BIN)audio-ops demo-data --stream-demo mediamtx/test_audio/demo.wav

run:
	$(BIN)audio-ops run $(TASK)

dev:
	@mkdir -p dagster_home
	DAGSTER_HOME=$${DAGSTER_HOME:-$(CURDIR)/dagster_home} $(BIN)audio-ops dev

serve:
	$(BIN)audio-ops serve --reload

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not integration"

lint:
	$(BIN)ruff check .

format:
	$(BIN)ruff check --fix .

typecheck:
	$(BIN)mypy audio_ops_dagster audio_ops_inference

docs:
	$(BIN)zensical build --clean

docs-serve:
	$(BIN)zensical serve

build:
	$(PY) -m build
	$(BIN)twine check dist/*

up:
	@test -f .env || cp env.example .env
	$(CORE) up -d

up-inference:
	@test -f .env || cp env.example .env
	$(INFER) up -d

up-all:
	@test -f .env || cp env.example .env
	$(ALL) up -d

down:
	$(ALL) down

logs:
	$(ALL) logs -f

ps:
	$(ALL) ps

images:
	$(ALL) build

new-task:
	@test -n "$(NAME)" || (echo "Usage: make new-task NAME=my_task [TYPE=classification]" && exit 1)
	$(BIN)audio-ops new-task $(NAME) $(if $(TYPE),--type $(TYPE),)

clean:
	rm -rf storage/* dagster_home/* mlruns/* site/ dist/ build/ .pytest_cache .ruff_cache
	@echo "Cleaned local runtime artifacts."
