"""
S3StorageBackend — Generic S3-compatible object storage for Audio Ops Framework.

Built on boto3 so it works transparently with:
  • MinIO  (local/on-prem)
  • AWS S3 (production cloud)
  • Cloudflare R2, Backblaze B2, or any S3-compatible service

Bucket key convention:
  {task_name}/audio/          ← raw audio inputs (optional upstream sync)
  {task_name}/features/       ← .npy mel spectrograms from AudioProcessor
  {task_name}/dataset/        ← train.csv / val.csv / test.csv from AudioDataset
  {task_name}/models/         ← best_model.pt checkpoints from ModelTrainer
  {task_name}/outputs/        ← inference results, evaluation JSON reports
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact type constants — keeps path construction consistent everywhere
# ─────────────────────────────────────────────────────────────────────────────
class ArtifactType:
    AUDIO    = "audio"
    FEATURES = "features"
    DATASET  = "dataset"
    MODELS   = "models"
    OUTPUTS  = "outputs"


class S3StorageBackend:
    """
    Thin, generic S3 wrapper used throughout the pipeline.

    Lazy-initialises the boto3 client on first use — this means importing this
    module never fails even when boto3 or MinIO is not reachable at startup.

    Parameters
    ----------
    endpoint_url  : Full URL including scheme, e.g. "http://minio:9000".
                    Leave None to use real AWS S3.
    access_key    : S3 / MinIO access key (MINIO_ROOT_USER or AWS_ACCESS_KEY_ID).
    secret_key    : S3 / MinIO secret key.
    bucket        : Default bucket name. Created automatically if it doesn't exist.
    region        : AWS-style region string (MinIO accepts any value here).
    use_ssl       : Whether to use HTTPS. Should be True in production.
    enabled       : When False every call is a no-op. Lets tasks run without
                    storage configured (useful in local dev / CI).
    """

    def __init__(
        self,
        endpoint_url: Optional[str],
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1",
        use_ssl: bool = False,
        enabled: bool = True,
    ):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.region = region
        self.use_ssl = use_ssl
        self.enabled = enabled

        self._client = None   # boto3 S3 client (lazy)
        self._bucket_ready: set[str] = set()  # buckets already confirmed to exist

    # ─── Client bootstrap ────────────────────────────────────────────────────

    def _get_client(self):
        """Return (and lazily create) the boto3 S3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for S3/MinIO storage. "
                    "Add it to pyproject.toml: boto3 = \"^1.34\""
                ) from exc

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                use_ssl=self.use_ssl,
                config=Config(
                    signature_version="s3v4",
                    # Retry up to 3 times with exponential back-off
                    retries={"max_attempts": 3, "mode": "adaptive"},
                ),
            )
            logger.debug("S3 client initialised (endpoint=%s)", self.endpoint_url)
        return self._client

    # ─── Bucket management ───────────────────────────────────────────────────

    def ensure_bucket(self, bucket: Optional[str] = None) -> None:
        """Create the bucket if it does not already exist (idempotent)."""
        if not self.enabled:
            return
        bucket = bucket or self.bucket
        if bucket in self._bucket_ready:
            return
        client = self._get_client()
        try:
            client.head_bucket(Bucket=bucket)
        except Exception:
            # Bucket not found — create it
            try:
                if self.region == "us-east-1":
                    client.create_bucket(Bucket=bucket)
                else:
                    client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.region},
                    )
                logger.info("Created bucket: %s", bucket)
            except Exception as exc:
                # Bucket may have been created by a concurrent process
                logger.warning("Could not create bucket %s: %s", bucket, exc)
        self._bucket_ready.add(bucket)

    # ─── Core file operations ─────────────────────────────────────────────────

    def upload_file(
        self,
        local_path: str | Path,
        key: str,
        bucket: Optional[str] = None,
        extra_args: Optional[dict] = None,
    ) -> str:
        """
        Upload a single file to S3/MinIO.

        Returns the full s3:// URI of the uploaded object.
        boto3 handles multipart automatically for files > 8 MB.
        """
        if not self.enabled:
            return f"s3://{bucket or self.bucket}/{key}  [storage disabled]"

        bucket = bucket or self.bucket
        self.ensure_bucket(bucket)
        client = self._get_client()

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Cannot upload — file not found: {local_path}")

        client.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=key,
            ExtraArgs=extra_args or {},
        )
        uri = f"s3://{bucket}/{key}"
        logger.debug("Uploaded %s → %s", local_path.name, uri)
        return uri

    def download_file(
        self,
        key: str,
        local_path: str | Path,
        bucket: Optional[str] = None,
    ) -> Path:
        """
        Download a single object to *local_path*.

        Parent directories are created automatically.
        Returns the resolved local Path so callers can use it immediately.
        """
        if not self.enabled:
            raise RuntimeError("Storage is disabled — cannot download files.")

        bucket = bucket or self.bucket
        client = self._get_client()

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=str(local_path),
        )
        logger.debug("Downloaded s3://%s/%s → %s", bucket, key, local_path)
        return local_path

    def upload_directory(
        self,
        local_dir: str | Path,
        prefix: str,
        bucket: Optional[str] = None,
        glob_pattern: str = "**/*",
    ) -> List[str]:
        """
        Recursively upload every file in *local_dir* under the given S3 *prefix*.

        Key structure:  prefix / relative_path_inside_local_dir
        Returns        list of s3:// URIs that were uploaded.

        Example:
            upload_directory("/tmp/features", "sentiment/features")
            → uploads  /tmp/features/positive/clip_seg_001.npy
              as        sentiment/features/positive/clip_seg_001.npy
        """
        if not self.enabled:
            return []

        bucket = bucket or self.bucket
        self.ensure_bucket(bucket)
        local_dir = Path(local_dir)

        if not local_dir.exists():
            logger.warning("upload_directory: local_dir does not exist: %s", local_dir)
            return []

        uploaded = []
        for file_path in local_dir.glob(glob_pattern):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_dir)
            key = f"{prefix}/{relative}".replace("\\", "/")  # Windows safety
            uri = self.upload_file(file_path, key, bucket=bucket)
            uploaded.append(uri)

        logger.info(
            "Uploaded %d files from %s → s3://%s/%s",
            len(uploaded), local_dir, bucket, prefix,
        )
        return uploaded

    def download_directory(
        self,
        prefix: str,
        local_dir: str | Path,
        bucket: Optional[str] = None,
    ) -> Path:
        """
        Download all objects under *prefix* into *local_dir*, recreating the
        sub-directory structure.

        Returns the resolved local directory Path.
        """
        if not self.enabled:
            raise RuntimeError("Storage is disabled — cannot download directories.")

        bucket = bucket or self.bucket
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        for key in self.list_objects(prefix, bucket=bucket):
            # Strip the prefix to get the relative path
            relative = key[len(prefix):].lstrip("/")
            dest = local_dir / relative
            self.download_file(key, dest, bucket=bucket)

        return local_dir

    # ─── Query / metadata ────────────────────────────────────────────────────

    def list_objects(
        self,
        prefix: str,
        bucket: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Yield all object keys under *prefix* (handles pagination automatically).
        """
        if not self.enabled:
            return

        bucket = bucket or self.bucket
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def object_exists(self, key: str, bucket: Optional[str] = None) -> bool:
        """Return True if the object exists in the bucket."""
        if not self.enabled:
            return False
        bucket = bucket or self.bucket
        try:
            self._get_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def get_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
        operation: str = "get_object",
    ) -> str:
        """
        Generate a pre-signed URL for sharing an object.

        *operation* is either "get_object" (download) or "put_object" (upload).
        *expires_in* is in seconds (default 1 hour).
        """
        if not self.enabled:
            raise RuntimeError("Storage is disabled.")
        bucket = bucket or self.bucket
        return self._get_client().generate_presigned_url(
            ClientMethod=operation,
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    # ─── High-level pipeline helpers ─────────────────────────────────────────

    def task_key(self, task_name: str, artifact_type: str, filename: str) -> str:
        """
        Build a canonical S3 key for a pipeline artifact.

        e.g. task_key("sentiment", ArtifactType.MODELS, "best_model.pt")
             → "sentiment/models/best_model.pt"
        """
        return f"{task_name}/{artifact_type}/{filename}"

    def upload_task_directory(
        self,
        task_name: str,
        artifact_type: str,
        local_dir: str | Path,
        bucket: Optional[str] = None,
    ) -> List[str]:
        """
        Upload an entire local directory as a specific artifact type for a task.

        Convenience wrapper around upload_directory that builds the prefix
        automatically from task_name + artifact_type.
        """
        prefix = f"{task_name}/{artifact_type}"
        return self.upload_directory(local_dir, prefix, bucket=bucket)

    def upload_task_file(
        self,
        task_name: str,
        artifact_type: str,
        local_path: str | Path,
        bucket: Optional[str] = None,
    ) -> str:
        """
        Upload a single file under the task/artifact_type prefix.

        The key is: {task_name}/{artifact_type}/{filename}
        """
        local_path = Path(local_path)
        key = self.task_key(task_name, artifact_type, local_path.name)
        return self.upload_file(local_path, key, bucket=bucket)

    def download_task_file(
        self,
        task_name: str,
        artifact_type: str,
        filename: str,
        local_dir: str | Path,
        bucket: Optional[str] = None,
    ) -> Path:
        """
        Download a specific file for a task/artifact_type into local_dir.
        """
        key = self.task_key(task_name, artifact_type, filename)
        local_path = Path(local_dir) / filename
        return self.download_file(key, local_path, bucket=bucket)

    # ─── Factory class methods ────────────────────────────────────────────────

    @classmethod
    def from_env(cls, bucket: Optional[str] = None) -> "S3StorageBackend":
        """
        Build an S3StorageBackend from standard environment variables.

        Reads:
          MINIO_ENDPOINT      — e.g. "http://minio:9000"
          MINIO_ACCESS_KEY    — root user / access key
          MINIO_SECRET_KEY    — root password / secret key
          MINIO_BUCKET_NAME   — default bucket
          MINIO_REGION        — default "us-east-1"
          MINIO_USE_SSL       — "true" / "false"
          ENABLE_MINIO        — "false" disables storage without errors
        """
        enabled = os.getenv("ENABLE_MINIO", "true").lower() != "false"

        endpoint_host = os.getenv("MINIO_HOST", "minio")
        endpoint_port = os.getenv("MINIO_PORT", "9000")
        use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
        scheme = "https" if use_ssl else "http"
        raw_endpoint = os.getenv("MINIO_ENDPOINT", "")
        if raw_endpoint and raw_endpoint.startswith(("http://", "https://")):
            endpoint_url = raw_endpoint
        else:
            endpoint_url = f"{scheme}://{endpoint_host}:{endpoint_port}"

        return cls(
            endpoint_url=endpoint_url,
            access_key=os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            bucket=bucket or os.getenv("MINIO_BUCKET_NAME", "audioops-bucket"),
            region=os.getenv("MINIO_REGION", "us-east-1"),
            use_ssl=use_ssl,
            enabled=enabled,
        )

    @classmethod
    def from_minio_config(cls, minio_config) -> "S3StorageBackend":
        """
        Build from a MinIOConfig dataclass instance (from config_loader.py).

        Usage:
            from config_loader import get_config
            backend = S3StorageBackend.from_minio_config(get_config().minio)
        """
        scheme = "https" if minio_config.use_ssl else "http"
        endpoint_url = f"{scheme}://{minio_config.host}:{minio_config.port}"
        return cls(
            endpoint_url=endpoint_url,
            access_key=minio_config.root_user,
            secret_key=minio_config.root_password,
            bucket=minio_config.bucket_name,
            region=minio_config.region,
            use_ssl=minio_config.use_ssl,
        )

    def __repr__(self) -> str:
        return (
            f"S3StorageBackend(endpoint={self.endpoint_url!r}, "
            f"bucket={self.bucket!r}, enabled={self.enabled})"
        )
