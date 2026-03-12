.PHONY: help setup config clean up logs validate secrets health

# Configuration
SHELL := /bin/bash
ENV_FILE := .env
SECRETS_FILE := .secrets.env
CONFIG_DIR := config
DOCKER_COMPOSE := docker-compose --env-file $(ENV_FILE)

help:
	@echo "Audio Ops Framework - Management Commands"
	@echo ""
	@echo "Setup & Configuration:"
	@echo "  make setup              - Initial setup (create directories, copy templates)"
	@echo "  make config             - Configure environment variables"
	@echo "  make secrets            - Configure secrets"
	@echo "  make validate           - Validate configuration"
	@echo ""
	@echo "Docker Compose:"
	@echo "  make up                 - Start all services"
	@echo "  make down               - Stop all services"
	@echo "  make logs               - Tail service logs"
	@echo "  make ps                 - Show running services"
	@echo ""
	@echo "Health & Status:"
	@echo "  make health             - Check service health"
	@echo "  make status             - Show service status"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - Remove containers and volumes"
	@echo "  make clean-all          - Remove everything (careful!)"
	@echo ""
	@echo "Development:"
	@echo "  make logs-dagster       - Tail Dagster logs"
	@echo "  make logs-inference     - Tail Inference logs"
	@echo "  make logs-influxdb      - Tail InfluxDB logs"
	@echo "  make shell-SERVICE      - Open shell in service"

# Setup
setup:
	@echo "Creating directory structure..."
	@mkdir -p $(CONFIG_DIR) dagster_home mlflow mlflow/artifacts label_studio_data minio_storage influxdb_data grafana_data demo_audio logs
	@echo "Creating configuration files..."
	@if [ ! -f $(ENV_FILE) ]; then cp .env.example $(ENV_FILE); echo "Created $(ENV_FILE)"; fi
	@if [ ! -f $(SECRETS_FILE) ]; then cp .secrets.env.example $(SECRETS_FILE); chmod 600 $(SECRETS_FILE); echo "Created $(SECRETS_FILE)"; fi
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit $(ENV_FILE) with your settings"
	@echo "  2. Edit $(SECRETS_FILE) with your credentials (keep it private!)"
	@echo "  3. Run 'make validate' to check configuration"
	@echo "  4. Run 'make up' to start services"

# Configuration
config:
	@echo "Opening configuration editor..."
	@if command -v nano &> /dev/null; then nano $(ENV_FILE); fi
	@if command -v vim &> /dev/null && ! command -v nano &> /dev/null; then vim $(ENV_FILE); fi

secrets:
	@echo "Opening secrets editor..."
	@chmod 600 $(SECRETS_FILE)
	@if command -v nano &> /dev/null; then nano $(SECRETS_FILE); fi
	@if command -v vim &> /dev/null && ! command -v nano &> /dev/null; then vim $(SECRETS_FILE); fi
	@echo ""
	@echo "⚠ Remember: Never commit $(SECRETS_FILE) to version control!"

# Validation
validate:
	@echo "Validating configuration..."
	@if [ ! -f $(ENV_FILE) ]; then echo "❌ $(ENV_FILE) not found"; exit 1; fi
	@echo "✓ $(ENV_FILE) found"
	@if [ ! -f $(SECRETS_FILE) ]; then echo "⚠ $(SECRETS_FILE) not found (using defaults)"; else echo "✓ $(SECRETS_FILE) found"; fi
	@echo ""
	@echo "Checking required variables..."
	@grep -q "MINIO_ROOT_USER" $(ENV_FILE) && echo "✓ MINIO_ROOT_USER" || echo "❌ MINIO_ROOT_USER"
	@grep -q "INFLUXDB_ADMIN_PASSWORD" $(ENV_FILE) && echo "✓ INFLUXDB_ADMIN_PASSWORD" || echo "❌ INFLUXDB_ADMIN_PASSWORD"
	@grep -q "GRAFANA_ADMIN_PASSWORD" $(ENV_FILE) && echo "✓ GRAFANA_ADMIN_PASSWORD" || echo "❌ GRAFANA_ADMIN_PASSWORD"
	@echo ""
	@echo "Configuration validation complete!"

# Docker Compose commands
up:
	@echo "Starting Audio Ops services..."
	@$(DOCKER_COMPOSE) up -d
	@echo ""
	@echo "Services starting up. Run 'make logs' to see output."

down:
	@echo "Stopping Audio Ops services..."
	@$(DOCKER_COMPOSE) down

logs:
	@$(DOCKER_COMPOSE) logs -f

ps:
	@$(DOCKER_COMPOSE) ps

logs-dagster:
	@$(DOCKER_COMPOSE) logs -f dagster

logs-inference:
	@$(DOCKER_COMPOSE) logs -f inference

logs-influxdb:
	@$(DOCKER_COMPOSE) logs -f influxdb

logs-mlflow:
	@$(DOCKER_COMPOSE) logs -f mlflow

logs-minio:
	@$(DOCKER_COMPOSE) logs -f minio

shell-%:
	@$(DOCKER_COMPOSE) exec $* /bin/bash

# Health checks
health:
	@echo "Checking service health..."
	@echo ""
	@echo "Dagster:"
	@curl -s http://localhost:3000/api/health || echo "❌ Not responding"
	@echo ""
	@echo "Inference:"
	@curl -s http://localhost:8000/health || echo "❌ Not responding"
	@echo ""
	@echo "InfluxDB:"
	@curl -s http://localhost:8086/health || echo "❌ Not responding"
	@echo ""
	@echo "MLFlow:"
	@curl -s http://localhost:5000/ > /dev/null && echo "✓ Running" || echo "❌ Not responding"
	@echo ""
	@echo "MinIO:"
	@curl -s http://localhost:9000/minio/health/live || echo "❌ Not responding"
	@echo ""
	@echo "Label Studio:"
	@curl -s http://localhost:8080/ > /dev/null && echo "✓ Running" || echo "❌ Not responding"
	@echo ""
	@echo "Grafana:"
	@curl -s http://localhost:3001/ > /dev/null && echo "✓ Running" || echo "❌ Not responding"

status: ps

# Cleanup
clean:
	@echo "Stopping and removing containers..."
	@$(DOCKER_COMPOSE) down
	@echo "✓ Containers removed"

clean-volumes:
	@echo "⚠ Removing volumes (this deletes data!)"
	@$(DOCKER_COMPOSE) down -v
	@echo "✓ Volumes removed"

clean-all:
	@echo "⚠⚠⚠ WARNING: Removing everything (containers, volumes, images)"
	@read -p "Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(DOCKER_COMPOSE) down -v --rmi all; \
		echo "✓ Everything removed"; \
	else \
		echo "Cancelled"; \
	fi

# Development commands
dev-setup: setup validate
	@echo "Development environment ready!"
	@echo ""
	@echo "Start services with: make up"

dev-logs:
	@$(DOCKER_COMPOSE) logs -f --tail=50

rebuild:
	@echo "Rebuilding Docker images..."
	@$(DOCKER_COMPOSE) build --no-cache
	@echo "✓ Build complete"

restart: down up
	@echo "Services restarted"

# Database commands
db-shell:
	@$(DOCKER_COMPOSE) exec influxdb influx

# Backup commands
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/audioops-backup-$$(date +%Y%m%d-%H%M%S).tar.gz \
		dagster_home mlflow/artifacts label_studio_data influxdb_data grafana_data
	@echo "✓ Backup created"

# Configuration export/import
export-config:
	@echo "Exporting configuration..."
	@python3 -c "from config_loader import get_config; config = get_config(); config.to_yaml('config/config-export.yaml'); print('✓ Exported to config/config-export.yaml')"

import-config:
	@echo "Importing configuration..."
	@python3 -c "from config_loader import AudioOpsConfig; config = AudioOpsConfig.from_yaml('config/config.yaml'); print('✓ Configuration loaded')"

# Port forwarding (for remote access)
port-forward:
	@echo "Setting up port forwarding..."
	@echo "Dagster: 3000"
	@echo "MLFlow: 5000"
	@echo "MinIO API: 9000"
	@echo "MinIO Console: 9001"
	@echo "InfluxDB: 8086"
	@echo "Inference: 8000"
	@echo "Label Studio: 8080"
	@echo "Grafana: 3001"

# Documentation
docs:
	@cat CONFIG_README.md | less

# Quick reference
quick:
	@echo "Audio Ops Framework - Quick Commands"
	@echo ""
	@echo "First time: make setup && make validate && make up"
	@echo "View logs: make logs"
	@echo "Check health: make health"
	@echo "Stop: make down"
	@echo "Clean: make clean-all"
	@echo ""
	@echo "For full help: make help"