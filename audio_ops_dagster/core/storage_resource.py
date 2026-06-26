"""
MinIOStorageResource — Dagster ConfigurableResource for S3/MinIO storage.

This is the glue between the Dagster dependency-injection system and the
raw S3StorageBackend. Assets declare it in required_resource_keys and call
context.resources.minio_storage.upload_task_directory(...) etc.

Having a real ConfigurableResource means:
  • Fields are visible and configurable in the Dagster UI
  • The resource can be swapped out in tests (just pass a different instance)
  • Dagster knows about its lifecycle (set-up / tear-down)

FRAMEWORK NOTE (local-first)
----------------------------
This template ships without an object-storage service, so the resource is
DISABLED by default — every call is a silent no-op and the pipeline keeps all
artifacts on the local filesystem. Turn it on with STORAGE_BACKEND=minio (or
ENABLE_MINIO=true) plus the MINIO_* env vars. The asset code only ever uploads
when a task config also sets `use_remote_storage: true`, so enabling the
backend alone changes nothing until a task opts in.
"""

from __future__ import annotations

import logging
import os
from functools import cached_property
from pathlib import Path
from typing import List, Optional

from dagster import ConfigurableResource

from audio_ops_dagster.core.storage import ArtifactType, S3StorageBackend

logger = logging.getLogger(__name__)


class MinIOStorageResource(ConfigurableResource):
    """
    Dagster resource that exposes an S3StorageBackend to pipeline assets.

    Fields
    ------
    endpoint_url  : Full URL to the MinIO / S3 endpoint.
    access_key    : Access key (maps to MINIO_ROOT_USER in .env).
    secret_key    : Secret key (maps to MINIO_ROOT_PASSWORD in .env).
    bucket        : Bucket name.  The bucket is created on first use.
    region        : S3 region (MinIO ignores this but boto3 requires it).
    use_ssl       : Enable HTTPS (set True in production).
    enabled       : When False all storage calls are silent no-ops. Default
                    False here (local-first); from_env() flips it on when
                    STORAGE_BACKEND=minio/s3 or ENABLE_MINIO=true.
    """

    endpoint_url: str = "http://minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "audioops-bucket"
    region: str = "us-east-1"
    use_ssl: bool = False
    enabled: bool = False

    # ── Internal backend (not part of the Dagster config schema) ─────────────

    def _get_backend(self) -> S3StorageBackend:
        """Lazily build (and cache) the underlying storage backend."""
        cache_key = id(self)
        if cache_key not in _backend_cache:
            _backend_cache[cache_key] = S3StorageBackend(
                endpoint_url=self.endpoint_url,
                access_key=self.access_key,
                secret_key=self.secret_key,
                bucket=self.bucket,
                region=self.region,
                use_ssl=self.use_ssl,
                enabled=self.enabled,
            )
        return _backend_cache[cache_key]

    # ── Convenience pass-throughs so assets don't need to import backend ─────

    def upload_file(self, local_path: str | Path, key: str) -> str:
        return self._get_backend().upload_file(local_path, key)

    def download_file(self, key: str, local_path: str | Path) -> Path:
        return self._get_backend().download_file(key, local_path)

    def upload_directory(self, local_dir: str | Path, prefix: str) -> List[str]:
        return self._get_backend().upload_directory(local_dir, prefix)

    def download_directory(self, prefix: str, local_dir: str | Path) -> Path:
        return self._get_backend().download_directory(prefix, local_dir)

    def object_exists(self, key: str) -> bool:
        return self._get_backend().object_exists(key)

    def list_objects(self, prefix: str):
        return self._get_backend().list_objects(prefix)

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self._get_backend().get_presigned_url(key, expires_in=expires_in)

    # ── High-level task-aware helpers ─────────────────────────────────────────

    def upload_task_directory(
        self,
        task_name: str,
        artifact_type: str,
        local_dir: str | Path,
    ) -> List[str]:
        """
        Upload an entire directory as a typed artifact for a named task.

        Example:
            storage.upload_task_directory("sentiment", ArtifactType.FEATURES,
                                          "/app/storage/sentiment/features")
            → uploads all .npy files under  sentiment/features/...  in MinIO
        """
        return self._get_backend().upload_task_directory(task_name, artifact_type, local_dir)

    def upload_task_file(
        self,
        task_name: str,
        artifact_type: str,
        local_path: str | Path,
    ) -> str:
        """
        Upload a single file as a typed artifact for a named task.

        Example:
            storage.upload_task_file("sentiment", ArtifactType.MODELS,
                                     "/app/storage/sentiment/checkpoints/best_model.pt")
            → key:  sentiment/models/best_model.pt
        """
        return self._get_backend().upload_task_file(task_name, artifact_type, local_path)

    def download_task_file(
        self,
        task_name: str,
        artifact_type: str,
        filename: str,
        local_dir: str | Path,
    ) -> Path:
        """
        Download a specific artifact file for a task.

        Example:
            path = storage.download_task_file(
                "sentiment", ArtifactType.MODELS, "best_model.pt",
                "/app/storage/sentiment/checkpoints"
            )
        """
        return self._get_backend().download_task_file(
            task_name, artifact_type, filename, local_dir
        )

    def ensure_bucket(self) -> None:
        """Ensure the configured bucket exists (called once at startup)."""
        self._get_backend().ensure_bucket()

    # ── Factory class method ──────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "MinIOStorageResource":
        """
        Build the resource entirely from environment variables.

        Local-first: storage is OFF unless STORAGE_BACKEND is minio/s3 or
        ENABLE_MINIO=true. Variable names match docker-compose.yml / env.example.
        """
        backend = os.getenv("STORAGE_BACKEND", "local").lower()
        enable_minio = os.getenv("ENABLE_MINIO", "").lower()
        enabled = backend in ("minio", "s3") or enable_minio == "true"

        use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
        scheme = "https" if use_ssl else "http"

        host = os.getenv("MINIO_HOST", "minio")
        port = os.getenv("MINIO_PORT", "9000")
        raw_endpoint = os.getenv("MINIO_ENDPOINT", "")
        if raw_endpoint and raw_endpoint.startswith(("http://", "https://")):
            endpoint_url = raw_endpoint
        else:
            endpoint_url = f"{scheme}://{host}:{port}"

        return cls(
            endpoint_url=endpoint_url,
            access_key=(
                os.getenv("MINIO_ACCESS_KEY")
                or os.getenv("MINIO_ROOT_USER", "minioadmin")
            ),
            secret_key=(
                os.getenv("MINIO_SECRET_KEY")
                or os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
            ),
            bucket=os.getenv("MINIO_BUCKET_NAME", "audioops-bucket"),
            region=os.getenv("MINIO_REGION", "us-east-1"),
            use_ssl=use_ssl,
            enabled=enabled,
        )


# Module-level cache so the backend is only created once per resource instance
_backend_cache: dict[int, S3StorageBackend] = {}
