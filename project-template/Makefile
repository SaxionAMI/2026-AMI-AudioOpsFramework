# Common commands for a project built on the Audio Ops Framework.
# `make setup` once, then the other targets. They use .venv automatically.
.PHONY: help setup train predict serve dev test list clean

BIN  := $(if $(wildcard .venv/bin/python),.venv/bin/,)
# The task to train or score; defaults to the first folder in tasks/.
TASK ?= $(firstword $(notdir $(patsubst %/config.yaml,%,$(wildcard tasks/*/config.yaml))))

# Only this project's tasks, not the framework's bundled example.
export AUDIO_OPS_DISABLE_BUILTIN_TASKS ?= true

help:
	@echo "  make setup                 one-time setup: .venv, PyTorch, framework, .env"
	@echo "  make train [TASK=name]     features, splits, training, evaluation, batch scoring"
	@echo "  make predict FILE=x.wav    score files with the trained model"
	@echo "  make serve                 inference API on http://localhost:8000/docs"
	@echo "  make dev                   Dagster UI on http://localhost:3000"
	@echo "  make test                  check the tasks against the installed framework"
	@echo "  make list                  tasks, capabilities and checkpoint status"
	@echo "  TASK is now: $(or $(TASK),(no task yet))"

setup:
	bash scripts/setup.sh

train:
	@test -n "$(TASK)" || (echo "No task yet: audio-ops new-task my_task --config-only" && exit 1)
	$(BIN)audio-ops run $(TASK)

predict:
	@test -n "$(FILE)" || (echo "usage: make predict FILE=clip.wav [TASK=name]" && exit 1)
	$(BIN)audio-ops predict $(TASK) $(FILE)

serve:
	$(BIN)audio-ops serve

dev:
	$(BIN)audio-ops dev

test:
	$(BIN)audio-ops validate
	@if [ -d tests ]; then $(BIN)python -m pytest -q; fi

list:
	$(BIN)audio-ops list

clean:
	rm -rf dagster_home/* storage/* mlruns/*
