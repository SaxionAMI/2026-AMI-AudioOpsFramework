# Audio Ops Framework - Configuration Guide

This document explains how to configure the Audio Ops Framework using environment variables, configuration files, and secrets management.

## Directory Structure

```
config/
├── config.yaml                 # Main configuration (YAML format)
├── config.json                 # Alternative configuration (JSON format)
├── config_minio.yaml           # MinIO storage configuration
├── config_mlflow.yaml          # MLFlow experiment tracking config
├── config_influxdb.yaml        # InfluxDB metrics configuration
├── config_inference.yaml       # Inference service configuration
├── .env.example                # Environment variables template
├── .secrets.env.example        # Secrets/credentials template
└── README.md                   # This file
```

## Quick Start

### 1. Create Environment Files

Copy the example files to your actual configuration files:

```bash
# Copy environment variables
cp .env.example .env

# Copy secrets (KEEP THIS PRIVATE!)
cp .secrets.env.example .secrets.env
chmod 600 .secrets.env  # Restrict permissions on Linux/Mac
```

### 2. Update Configuration

Edit `.env` and `.secrets.env` with your actual values:

```bash
# Edit environment variables
nano .env

# Edit secrets (never commit this file!)
nano .secrets.env
```

### 3. Update docker-compose.yml

Update your `docker-compose.yml` to use environment variables:

```yaml
services:
  minio:
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
  
  influxdb:
    environment:
      DOCKER_INFLUXDB_INIT_USERNAME: ${INFLUXDB_ADMIN_USER}
      DOCKER_INFLUXDB_INIT_PASSWORD: ${INFLUXDB_ADMIN_PASSWORD}
      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: ${INFLUXDB_ADMIN_TOKEN}
```

### 4. Start Services with Configuration

```bash
# Load environment and start
docker-compose --env-file .env up -d
```

## Configuration Methods

### Method 1: Environment Variables (.env)

**Best for:** Local development and simple deployments

1. Copy `.env.example` to `.env`
2. Update values in `.env`
3. Load with: `export $(cat .env | grep -v '#' | xargs)`

**Advantages:**
- Simple and familiar
- Works with docker-compose easily
- Override individual variables easily

**Disadvantages:**
- Large numbers of variables can be overwhelming
- No validation
- Flat structure

### Method 2: YAML Configuration (config.yaml)

**Best for:** Complex configurations and documentation

1. Edit `config/config.yaml`
2. Load in Python with `config_loader.py`:

```python
from config_loader import AudioOpsConfig

# Load from YAML
config = AudioOpsConfig.from_yaml("config/config.yaml")

# Access configuration
dagster_port = config.dagster.port
mlflow_uri = config.mlflow.tracking_uri
```

**Advantages:**
- Hierarchical structure
- Better documentation
- Type-safe in Python
- Easy to validate

**Disadvantages:**
- Requires parsing
- Need Python utilities

### Method 3: Environment Variables + YAML

**Best for:** Production deployments

Combine both for flexibility:

```python
# YAML provides structure, env vars override for deployment-specific values
config = AudioOpsConfig.from_yaml("config/config.yaml")
# Environment variables automatically override during load
```

## Secrets Management

### Never Commit Secrets!

Add to `.gitignore`:

```
.env
.secrets.env
.secrets/
secrets/
*.key
*.pem
```

### Using SecretsManager

Python-based secrets management:

```python
from secrets_manager import get_secrets_manager

secrets = get_secrets_manager()

# Get specific credentials
minio_creds = secrets.get_minio_credentials()
influxdb_creds = secrets.get_influxdb_credentials()

# Validate production secrets
if secrets.is_production():
    secrets.check_production_secrets()
```

### Docker Secrets (Production)

For production Docker deployments:

```yaml
services:
  minio:
    environment:
      MINIO_ROOT_PASSWORD_FILE: /run/secrets/minio_password
    secrets:
      - minio_password

secrets:
  minio_password:
    file: ./secrets/minio_password.txt
```

## Service-Specific Configuration

### MinIO Configuration

Edit `config/config_minio.yaml`:

```yaml
minio:
  root_user: minioadmin
  root_password: your-secure-password  # CHANGE THIS!
  bucket_name: audioops-bucket
  service_accounts:
    - username: audioops-user
      password: secure-password
```

### InfluxDB Configuration

Edit `config/config_influxdb.yaml`:

```yaml
influxdb:
  admin_password: admin123  # CHANGE THIS!
  retention_policies:
    - name: audio_metrics
      duration: 30d
```

### MLFlow Configuration

Edit `config/config_mlflow.yaml`:

```yaml
mlflow:
  backend_store_uri: sqlite:////mlflow/mlflow.db
  # Or use PostgreSQL:
  # backend_store_uri: postgresql://user:pass@postgres:5432/mlflow
```

### Inference Configuration

Edit `config/config_inference.yaml`:

```yaml
inference:
  models:
    sentiment_analysis:
      enabled: true
      model_path: /app/dagster_output/sentiment_model
```

## Environment-Specific Configuration

### Development (.env.dev)

```bash
APP_ENV=development
DAGSTER_LOG_LEVEL=DEBUG
INFERENCE_RELOAD=true
ENABLE_DEBUG_ENDPOINTS=true
```

### Production (.env.prod)

```bash
APP_ENV=production
DAGSTER_LOG_LEVEL=INFO
INFERENCE_RELOAD=false
ENABLE_DEBUG_ENDPOINTS=false
SECRET_KEY=your-very-secure-key
```

### Load Environment-Specific Config

```bash
# Load development config
docker-compose --env-file .env.dev up

# Load production config
docker-compose --env-file .env.prod up
```

## Configuration Validation

### Check Configuration

```python
from config_loader import get_config

# Load and validate
config = get_config()

# Validate required fields
assert config.dagster.port > 0
assert config.mlflow.tracking_uri
print("Configuration valid!")
```

### Validate Secrets

```python
from secrets_manager import get_secrets_manager

secrets = get_secrets_manager()

# Check production requirements
if secrets.is_production():
    secrets.check_production_secrets()
```

## Configuration Loading Priority

The system loads configuration in this order (later overrides earlier):

1. **Defaults** in code (lowest priority)
2. **YAML/JSON config files** (`config.yaml`)
3. **Environment variables** (`.env`)
4. **Secrets** (`.secrets.env`)
5. **Command-line arguments** (highest priority, if supported)

## Best Practices

### 1. Use YAML for Structure, Env Vars for Overrides

```yaml
# config.yaml
dagster:
  port: 3000
  host: 0.0.0.0
```

```bash
# .env (override for specific deployment)
DAGSTER_PORT=3001
```

### 2. Validate Configuration at Startup

```python
def startup():
    config = get_config()
    secrets = get_secrets_manager()
    
    # Validate
    assert config.dagster.port > 0
    if is_production():
        secrets.check_production_secrets()
```

### 3. Use Type Hints

```python
from config_loader import AudioOpsConfig, DagsterConfig

def initialize_dagster(config: DagsterConfig):
    """Initialize Dagster with validated configuration"""
    pass
```

### 4. Log Configuration (Masked)

```python
import json
from secrets_manager import get_secrets_manager

secrets = get_secrets_manager()

# Log with sensitive values masked
logger.info(f"Config: {json.dumps(secrets.to_dict(mask_sensitive=True))}")
```

### 5. Document Configuration Changes

When updating configuration, document:
- What changed
- Why it changed
- When it takes effect (requires restart? hot-reload?)
- How to rollback

## Troubleshooting

### Configuration Not Loading

```bash
# Check environment variables are loaded
echo $MINIO_ROOT_USER

# Check file exists
ls -la .env .secrets.env

# Check file format
cat .env | grep -v '^#' | grep -v '^$'
```

### Docker Container Can't Access Configuration

```bash
# Ensure volume mounts in docker-compose.yml
volumes:
  - .env:/app/.env
  - .secrets.env:/app/.secrets.env

# Or use env_file
env_file:
  - .env
  - .secrets.env
```

### Secrets Not Accessible in Python

```python
# Verify secrets manager initialized
from secrets_manager import get_secrets_manager
secrets = get_secrets_manager()
print(f"Secrets loaded: {secrets.secrets_loaded}")

# Check file path
import os
print(f".secrets.env exists: {os.path.exists('.secrets.env')}")
```

## Advanced Configuration

### Multi-Instance Deployment

Create separate config files:

```
config/
├── instances/
│  ├── instance1.yaml
│  ├── instance2.yaml
│  └── instance3.yaml
```

Load specific instance:

```bash
export CONFIG_INSTANCE=instance1
docker-compose up
```

### Configuration as Code

Use Python to generate configuration:

```python
from config_loader import AudioOpsConfig, DagsterConfig

# Generate configuration programmatically
config = AudioOpsConfig()
config.dagster = DagsterConfig(port=3000, host="0.0.0.0")

# Export for use
config.to_yaml("config/config.yaml")
```

### Integration with HashiCorp Vault

For enterprise secret management:

```python
import hvac

client = hvac.Client(url='http://localhost:8200')
secret = client.secrets.kv.read_secret_version(path='audioops')

from secrets_manager import get_secrets_manager
# Populate secrets from Vault
```

## Useful Commands

```bash
# Create required directories
mkdir -p config dagster_home mlflow mlflow/artifacts label_studio_data minio_storage influxdb_data grafana_data demo_audio logs

# Set up environment
cp .env.example .env
cp .secrets.env.example .secrets.env
chmod 600 .secrets.env

# Validate configuration
python -c "from config_loader import get_config; config = get_config(); print('Config valid!')"

# Validate secrets
python -c "from secrets_manager import get_secrets_manager; s = get_secrets_manager(); print(f'Secrets loaded: {s.secrets_loaded}')"

# Start services
docker-compose --env-file .env up -d

# Check logs
docker-compose logs -f dagster

# Stop services
docker-compose down
```

## Support and Documentation

- **Config Loader**: See `config_loader.py` docstrings
- **Secrets Manager**: See `secrets_manager.py` docstrings
- **Framework Docs**: Check main README.md