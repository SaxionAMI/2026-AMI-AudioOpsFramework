"""
Secrets Manager for Audio Ops Framework
Handles secure credential loading and management
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import json


class SecretsManager:
    """Manage secrets and sensitive configuration"""
    
    def __init__(self, secrets_file: Optional[str] = None):
        """
        Initialize secrets manager
        
        Args:
            secrets_file: Path to .secrets.env file (default: .secrets.env)
        """
        self.secrets_file = secrets_file or ".secrets.env"
        self.secrets_loaded = False
        self._secrets_cache: Dict[str, Any] = {}
        
        # Load secrets
        if Path(self.secrets_file).exists():
            load_dotenv(self.secrets_file)
            self.secrets_loaded = True
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value
        
        Args:
            key: Secret key name
            default: Default value if not found
        
        Returns:
            Secret value or default
        
        Raises:
            ValueError: If secret not found and no default provided
        """
        # Check cache first
        if key in self._secrets_cache:
            return self._secrets_cache[key]
        
        # Get from environment
        value = os.getenv(key)
        
        if value is None and default is None:
            raise ValueError(f"Secret '{key}' not found and no default provided")
        
        # Cache the value
        self._secrets_cache[key] = value or default
        return value or default
    
    def get_minio_credentials(self) -> Dict[str, str]:
        """Get MinIO credentials"""
        return {
            "access_key": self.get_secret("MINIO_ROOT_USER", "minioadmin"),
            "secret_key": self.get_secret("MINIO_ROOT_PASSWORD", "minioadmin"),
            "service_user": self.get_secret("MINIO_SERVICE_ACCOUNT_USER", "audioops-user"),
            "service_password": self.get_secret("MINIO_SERVICE_ACCOUNT_PASSWORD"),
        }
    
    def get_influxdb_credentials(self) -> Dict[str, str]:
        """Get InfluxDB credentials"""
        return {
            "username": self.get_secret("INFLUXDB_ADMIN_USER", "admin"),
            "password": self.get_secret("INFLUXDB_ADMIN_PASSWORD"),
            "token": self.get_secret("INFLUXDB_ADMIN_TOKEN"),
        }
    
    def get_mlflow_uri(self) -> str:
        """Get MLFlow backend URI"""
        return self.get_secret(
            "MLFLOW_BACKEND_URI",
            "sqlite:////mlflow/mlflow.db"
        )
    
    def get_labelstudio_credentials(self) -> Dict[str, str]:
        """Get Label Studio credentials"""
        return {
            "password": self.get_secret("LABELSTUDIO_PASSWORD"),
            "api_key": self.get_secret("LABELSTUDIO_API_KEY"),
        }
    
    def get_grafana_credentials(self) -> Dict[str, str]:
        """Get Grafana credentials"""
        return {
            "admin_password": self.get_secret("GRAFANA_ADMIN_PASSWORD"),
            "api_token": self.get_secret("GRAFANA_API_TOKEN"),
        }
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get API keys and tokens"""
        return {
            "secret_key": self.get_secret("API_SECRET_KEY"),
            "jwt_secret": self.get_secret("JWT_SECRET"),
            "dagster_token": self.get_secret("DAGSTER_API_TOKEN"),
        }
    
    def get_aws_credentials(self) -> Dict[str, str]:
        """Get AWS credentials"""
        return {
            "access_key_id": self.get_secret("AWS_ACCESS_KEY_ID"),
            "secret_access_key": self.get_secret("AWS_SECRET_ACCESS_KEY"),
            "region": self.get_secret("AWS_REGION", "us-east-1"),
        }
    
    def get_database_uri(self, db_type: str = "postgres") -> str:
        """
        Get database connection URI
        
        Args:
            db_type: Type of database (postgres, mysql, etc.)
        
        Returns:
            Database URI
        """
        if db_type == "postgres":
            user = self.get_secret("POSTGRES_USER")
            password = self.get_secret("POSTGRES_PASSWORD")
            host = self.get_secret("POSTGRES_HOST", "localhost")
            port = self.get_secret("POSTGRES_PORT", "5432")
            db = self.get_secret("POSTGRES_DB", "audioops")
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        
        elif db_type == "mysql":
            user = self.get_secret("MYSQL_USER")
            password = self.get_secret("MYSQL_PASSWORD")
            host = self.get_secret("MYSQL_HOST", "localhost")
            port = self.get_secret("MYSQL_PORT", "3306")
            db = self.get_secret("MYSQL_DB", "audioops")
            return f"mysql://{user}:{password}@{host}:{port}/{db}"
        
        else:
            raise ValueError(f"Unknown database type: {db_type}")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        host = self.get_secret("REDIS_HOST", "localhost")
        port = self.get_secret("REDIS_PORT", "6379")
        password = self.get_secret("REDIS_PASSWORD")
        
        if password:
            return f"redis://:{password}@{host}:{port}"
        return f"redis://{host}:{port}"
    
    def validate_required_secrets(self, required_keys: list) -> bool:
        """
        Validate that required secrets are present
        
        Args:
            required_keys: List of required secret keys
        
        Returns:
            True if all required secrets are present
        
        Raises:
            ValueError: If any required secret is missing
        """
        missing = []
        for key in required_keys:
            if not os.getenv(key):
                missing.append(key)
        
        if missing:
            raise ValueError(
                f"Missing required secrets: {', '.join(missing)}. "
                f"Please set them in {self.secrets_file}"
            )
        
        return True
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("APP_ENV", "development").lower() == "production"
    
    def check_production_secrets(self) -> None:
        """Check that production secrets are configured"""
        if self.is_production():
            required = [
                "MINIO_ROOT_PASSWORD",
                "INFLUXDB_ADMIN_PASSWORD",
                "GRAFANA_ADMIN_PASSWORD",
                "API_SECRET_KEY",
            ]
            
            if not self.validate_required_secrets(required):
                raise RuntimeError(
                    "Production mode requires all critical secrets to be set"
                )
    
    def clear_cache(self) -> None:
        """Clear the secrets cache"""
        self._secrets_cache.clear()
    
    @staticmethod
    def create_secrets_template() -> None:
        """Create a .secrets.env template file"""
        template_path = Path(".secrets.env.example")
        if not template_path.exists():
            raise FileNotFoundError(
                ".secrets.env.example not found. "
                "Please ensure the template exists."
            )
        
        secrets_path = Path(".secrets.env")
        if secrets_path.exists():
            print(f"Warning: {secrets_path} already exists")
            return
        
        # Copy template to .secrets.env
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        with open(secrets_path, 'w') as f:
            f.write(template_content)
        
        # Set restrictive permissions
        secrets_path.chmod(0o600)
        print(f"Created {secrets_path} with restricted permissions (600)")
    
    def to_dict(self, mask_sensitive: bool = True) -> Dict[str, str]:
        """
        Export secrets as dictionary
        
        Args:
            mask_sensitive: Mask sensitive values for logging
        
        Returns:
            Dictionary of secrets
        """
        result = {}
        sensitive_keys = [
            "PASSWORD", "KEY", "TOKEN", "SECRET", "CREDENTIAL"
        ]
        
        for key, value in self._secrets_cache.items():
            if mask_sensitive and any(s in key.upper() for s in sensitive_keys):
                result[key] = "***MASKED***"
            else:
                result[key] = value
        
        return result


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(secrets_file: Optional[str] = None) -> SecretsManager:
    """Get or create the global secrets manager"""
    global _secrets_manager
    
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(secrets_file)
    
    return _secrets_manager


def reset_secrets_manager() -> None:
    """Reset the global secrets manager"""
    global _secrets_manager
    _secrets_manager = None


# Convenience functions
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value"""
    return get_secrets_manager().get_secret(key, default)


def get_minio_credentials() -> Dict[str, str]:
    """Get MinIO credentials"""
    return get_secrets_manager().get_minio_credentials()


def get_influxdb_credentials() -> Dict[str, str]:
    """Get InfluxDB credentials"""
    return get_secrets_manager().get_influxdb_credentials()


if __name__ == "__main__":
    # Example usage
    try:
        secrets = get_secrets_manager()
        print("Secrets Manager Initialized")
        print(f"Secrets loaded: {secrets.secrets_loaded}")
        
        # Test getting specific credentials
        try:
            minio_creds = secrets.get_minio_credentials()
            print(f"MinIO credentials available: {bool(minio_creds.get('access_key'))}")
        except ValueError as e:
            print(f"Note: {e}")
        
    except Exception as e:
        print(f"Error initializing secrets: {e}")