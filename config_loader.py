"""
Configuration loader for Audio Ops Framework
Supports YAML config files and environment variable overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from dotenv import load_dotenv


@dataclass
class DagsterConfig:
    host: str = "0.0.0.0"
    port: int = 3000
    home: str = "/app/dagster_home"
    log_level: str = "INFO"


@dataclass
class MLFlowConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    tracking_uri: str = "http://mlflow:5000"
    backend_store_uri: str = "sqlite:////mlflow/mlflow.db"
    artifact_root: str = "/mlflow/artifacts"
    log_level: str = "INFO"


@dataclass
class MinIOConfig:
    host: str = "minio"
    port: int = 9000
    console_port: int = 9001
    root_user: str = "minioadmin"
    root_password: str = "minioadmin"
    bucket_name: str = "audioops-bucket"
    region: str = "us-east-1"
    use_ssl: bool = False


@dataclass
class InfluxDBConfig:
    host: str = "influxdb"
    port: int = 8086
    organization: str = "audioops"
    bucket: str = "audio_metrics"
    username: str = "admin"
    password: str = "admin123"
    auth_token: str = "audioopstoken"
    init_mode: str = "setup"
    retention_policy: str = "30d"


@dataclass
class InferenceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    log_level: str = "INFO"
    model_cache_dir: str = "/app/dagster_output"
    demo_audio_dir: str = "/app/demo_audio"


@dataclass
class LabelStudioConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    data_dir: str = "/label-studio/data"
    ml_timeout_predict: int = 3600
    ml_timeout_health: int = 60
    ml_timeout_train: int = 7200
    gunicorn_workers: int = 2
    gunicorn_timeout: int = 36000


@dataclass
class GrafanaConfig:
    host: str = "0.0.0.0"
    port: int = 3001
    admin_user: str = "admin"
    admin_password: str = "admin"
    allow_embedding: bool = True
    disable_sanitize_html: bool = True
    data_dir: str = "/var/lib/grafana"


@dataclass
class StorageConfig:
    dagster_home: str = "./dagster_home"
    dagster_output: str = "./dagster_output"
    mlflow: str = "./mlflow"
    mlflow_artifacts: str = "./mlflow/artifacts"
    label_studio_data: str = "./label_studio_data"
    minio_data: str = "./minio_storage"
    influxdb_data: str = "./influxdb_data"
    grafana_data: str = "./grafana_data"
    demo_audio: str = "./demo_audio"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/audioops.log"
    max_size_mb: int = 100
    backup_count: int = 5


@dataclass
class SecurityConfig:
    secret_key: str = "change-me-in-production"
    allowed_origins: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080"
    ])


@dataclass
class AudioOpsConfig:
    """Main configuration container for Audio Ops Framework"""
    app: Dict[str, Any] = field(default_factory=dict)
    dagster: DagsterConfig = field(default_factory=DagsterConfig)
    mlflow: MLFlowConfig = field(default_factory=MLFlowConfig)
    minio: MinIOConfig = field(default_factory=MinIOConfig)
    influxdb: InfluxDBConfig = field(default_factory=InfluxDBConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    labelstudio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    grafana: GrafanaConfig = field(default_factory=GrafanaConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "AudioOpsConfig":
        """Load configuration from YAML file"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, config_path: str) -> "AudioOpsConfig":
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls._from_dict(data)

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "AudioOpsConfig":
        """Load configuration from environment variables"""
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        elif Path(".env").exists():
            load_dotenv(".env")
        
        config = cls()
        
        # Override with environment variables
        config.app = {
            "name": os.getenv("APP_NAME", "audio-ops-framework"),
            "version": os.getenv("APP_VERSION", "1.0.0"),
            "environment": os.getenv("APP_ENV", "development"),
        }
        
        config.dagster = DagsterConfig(
            host=os.getenv("DAGSTER_HOST", "0.0.0.0"),
            port=int(os.getenv("DAGSTER_PORT", "3000")),
            home=os.getenv("DAGSTER_HOME", "/app/dagster_home"),
            log_level=os.getenv("DAGSTER_LOG_LEVEL", "INFO"),
        )
        
        config.mlflow = MLFlowConfig(
            host=os.getenv("MLFLOW_HOST", "0.0.0.0"),
            port=int(os.getenv("MLFLOW_PORT", "5000")),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            backend_store_uri=os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:////mlflow/mlflow.db"),
            artifact_root=os.getenv("MLFLOW_ARTIFACT_ROOT", "/mlflow/artifacts"),
            log_level=os.getenv("MLFLOW_LOG_LEVEL", "INFO"),
        )
        
        config.minio = MinIOConfig(
            host=os.getenv("MINIO_HOST", "minio"),
            port=int(os.getenv("MINIO_PORT", "9000")),
            console_port=int(os.getenv("MINIO_CONSOLE_PORT", "9001")),
            root_user=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            root_password=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            bucket_name=os.getenv("MINIO_BUCKET_NAME", "audioops-bucket"),
            region=os.getenv("MINIO_REGION", "us-east-1"),
            use_ssl=os.getenv("MINIO_USE_SSL", "false").lower() == "true",
        )
        
        config.influxdb = InfluxDBConfig(
            host=os.getenv("INFLUXDB_HOST", "influxdb"),
            port=int(os.getenv("INFLUXDB_PORT", "8086")),
            organization=os.getenv("INFLUXDB_INIT_ORG", "audioops"),
            bucket=os.getenv("INFLUXDB_INIT_BUCKET", "audio_metrics"),
            username=os.getenv("INFLUXDB_ADMIN_USER", "admin"),
            password=os.getenv("INFLUXDB_ADMIN_PASSWORD", "admin123"),
            auth_token=os.getenv("INFLUXDB_ADMIN_TOKEN", "audioopstoken"),
            init_mode=os.getenv("DOCKER_INFLUXDB_INIT_MODE", "setup"),
            retention_policy=os.getenv("INFLUXDB_RETENTION_POLICY", "30d"),
        )
        
        config.inference = InferenceConfig(
            host=os.getenv("INFERENCE_HOST", "0.0.0.0"),
            port=int(os.getenv("INFERENCE_PORT", "8000")),
            workers=int(os.getenv("INFERENCE_WORKERS", "4")),
            reload=os.getenv("INFERENCE_RELOAD", "true").lower() == "true",
            log_level=os.getenv("INFERENCE_LOG_LEVEL", "INFO"),
            model_cache_dir=os.getenv("INFERENCE_MODEL_CACHE", "/app/dagster_output"),
            demo_audio_dir=os.getenv("INFERENCE_DEMO_AUDIO", "/app/demo_audio"),
        )
        
        config.labelstudio = LabelStudioConfig(
            host=os.getenv("LABELSTUDIO_HOST", "0.0.0.0"),
            port=int(os.getenv("LABELSTUDIO_PORT", "8080")),
            data_dir=os.getenv("LABELSTUDIO_DATA_DIR", "/label-studio/data"),
            ml_timeout_predict=int(os.getenv("ML_TIMEOUT_PREDICT", "3600")),
            ml_timeout_health=int(os.getenv("ML_TIMEOUT_HEALTH", "60")),
            ml_timeout_train=int(os.getenv("ML_TIMEOUT_TRAIN", "7200")),
            gunicorn_workers=int(os.getenv("GUNICORN_WORKERS", "2")),
            gunicorn_timeout=int(os.getenv("GUNICORN_TIMEOUT", "36000")),
        )
        
        config.grafana = GrafanaConfig(
            host=os.getenv("GRAFANA_HOST", "0.0.0.0"),
            port=int(os.getenv("GRAFANA_PORT", "3001")),
            admin_user=os.getenv("GRAFANA_ADMIN_USER", "admin"),
            admin_password=os.getenv("GRAFANA_ADMIN_PASSWORD", "admin"),
            allow_embedding=os.getenv("GF_SECURITY_ALLOW_EMBEDDING", "true").lower() == "true",
            disable_sanitize_html=os.getenv("GF_PANELS_DISABLE_SANITIZE_HTML", "true").lower() == "true",
            data_dir=os.getenv("GRAFANA_DATA_DIR", "/var/lib/grafana"),
        )
        
        config.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "json"),
            file=os.getenv("LOG_FILE", "logs/audioops.log"),
            max_size_mb=int(os.getenv("LOG_MAX_SIZE_MB", "100")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        )
        
        config.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", "change-me-in-production"),
            allowed_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(","),
        )
        
        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AudioOpsConfig":
        """Create config from dictionary"""
        config = cls()
        
        if "app" in data:
            config.app = data["app"]
        
        if "dagster" in data:
            config.dagster = DagsterConfig(**data["dagster"])
        
        if "mlflow" in data:
            config.mlflow = MLFlowConfig(**data["mlflow"])
        
        if "minio" in data:
            config.minio = MinIOConfig(**data["minio"])
        
        if "influxdb" in data:
            config.influxdb = InfluxDBConfig(**data["influxdb"])
        
        if "inference" in data:
            config.inference = InferenceConfig(**data["inference"])
        
        if "labelstudio" in data:
            config.labelstudio = LabelStudioConfig(**data["labelstudio"])
        
        if "grafana" in data:
            config.grafana = GrafanaConfig(**data["grafana"])
        
        if "storage" in data:
            config.storage = StorageConfig(**data["storage"])
        
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        
        if "security" in data:
            config.security = SecurityConfig(**data["security"])
        
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "app": self.app,
            "dagster": asdict(self.dagster),
            "mlflow": asdict(self.mlflow),
            "minio": asdict(self.minio),
            "influxdb": asdict(self.influxdb),
            "inference": asdict(self.inference),
            "labelstudio": asdict(self.labelstudio),
            "grafana": asdict(self.grafana),
            "storage": asdict(self.storage),
            "logging": asdict(self.logging),
            "security": asdict(self.security),
        }

    def to_json(self, path: str) -> None:
        """Export config to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, path: str) -> None:
        """Export config to YAML file"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Global config instance
_config: Optional[AudioOpsConfig] = None


def get_config(force_reload: bool = False) -> AudioOpsConfig:
    """Get the global config instance (singleton pattern)"""
    global _config
    
    if _config is None or force_reload:
        # Try to load from YAML first, then fall back to env
        config_path = Path("config/config.yaml")
        if config_path.exists():
            _config = AudioOpsConfig.from_yaml(str(config_path))
        else:
            _config = AudioOpsConfig.from_env()
    
    return _config


def set_config(config: AudioOpsConfig) -> None:
    """Set the global config instance"""
    global _config
    _config = config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print("Audio Ops Configuration Loaded")
    print(f"App: {config.app}")
    print(f"Dagster: {config.dagster}")
    print(f"MLFlow URI: {config.mlflow.tracking_uri}")
    print(f"MinIO User: {config.minio.root_user}")