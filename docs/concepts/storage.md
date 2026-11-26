# Storage

## Local layout (default)

Everything a task produces lives under its `output_dir`
(default `storage/<task>/`):

```text
storage/<task>/
├── features/<label>/<recording>_seg_NNN.npy
├── features/_features_manifest.json
├── dataset/{train,val,test}.csv
├── checkpoints/best_model.pt          # weights-only, served by the API
├── checkpoints/last_checkpoint.pt     # full training state, for resume
└── outputs/
    ├── evaluation_<task>.json
    └── inference_results_<task>.{json,csv}
```

No service is needed. Relative paths resolve against the working directory of
the process (Dagster / API), which in the Docker images is `/app`.

## Object storage (opt-in)

One taxonomy (`core/storage.py` → `ArtifactType`): `audio`, `features`,
`dataset`, `models`, `outputs`. Keys are `{prefix}/{artifact_type}/{file}`,
where `prefix` is `storage_bucket_prefix` or the task name.

Enable with `STORAGE_BACKEND=minio` (or `s3`) plus the `MINIO_*` variables
([environment reference](../reference/environment.md#storage)); install the
`s3` extra. The boto3-based backend works with MinIO, AWS S3, Cloudflare R2 and
Backblaze B2. Uploads still only happen for tasks with `use_remote_storage: true`.

With the backend enabled, the inference service downloads
`{prefix}/models/best_model.pt` into `MODELS_DIR/<task>/` when no local
checkpoint exists, so training and serving can run on different machines.

!!! note "Credentials"
    No credentials are baked in. Leave `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY`
    unset on AWS to use boto3's default chain (env vars, instance roles, SSO).

## Checkpoint safety

`best_model.pt` contains only tensors and plain metadata and is always loaded
with `torch.load(weights_only=True)`. Pickled full-model files are refused
unless `ALLOW_UNSAFE_CHECKPOINTS=true`, because unpickling runs code from the file.
