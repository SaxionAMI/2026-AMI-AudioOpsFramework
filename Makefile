# Audio Ops Framework — common commands.
.PHONY: help setup up up-inference up-all down logs ps build dev test new-task clean

CORE    := docker compose
INFER   := docker compose -f docker-compose.inference.yml
ALL     := docker compose -f docker-compose.yml -f docker-compose.inference.yml

help:
	@echo "Audio Ops Framework"
	@echo "  make setup         Create runtime dirs and a .env from env.example"
	@echo "  make up            Start CORE (Dagster) — training & batch inference"
	@echo "  make up-inference  Start the streaming inference service only"
	@echo "  make up-all        Start core + inference together"
	@echo "  make down          Stop everything"
	@echo "  make logs / ps     Tail logs / list services (core)"
	@echo "  make build         Rebuild images (core + inference)"
	@echo "  make dev           Run Dagster locally (no Docker)"
	@echo "  make test          Run the test suite (pytest)"
	@echo "  make new-task      Scaffold a task: make new-task NAME=my_task [TYPE=regression]"
	@echo "  make clean         Remove local runtime artifacts"
	@echo ""
	@echo "  GPU: append -f docker-compose.gpu.yml to any compose command,"
	@echo "       e.g. docker compose -f docker-compose.inference.yml -f docker-compose.gpu.yml up -d"

setup:
	@mkdir -p storage audio_input dagster_home models mlruns audio_input/example_classifier
	@test -f .env || cp env.example .env
	@echo "Ready. Put audio under audio_input/example_classifier/<label>/*.wav"

up:
	$(CORE) up -d

up-inference:
	$(INFER) up -d

up-all:
	$(ALL) up -d

down:
	$(ALL) down

logs:
	$(CORE) logs -f

ps:
	$(ALL) ps

build:
	$(ALL) build

dev:
	dagster dev -w workspace.yaml

test:
	python -m pytest tests/ -q

new-task:
	@test -n "$(NAME)" || (echo "Usage: make new-task NAME=my_task [TYPE=classification]" && exit 1)
	python scripts/new_task.py $(NAME) $(if $(TYPE),--type $(TYPE),)

clean:
	rm -rf storage/* dagster_home/* mlruns/* models/*
	@echo "Cleaned local runtime artifacts."
