"""S3-compatible object store client (Constitution Art. XII §8).

Thin wrapper over `s3fs` (already an indirect dependency via `spatialdata`,
so no new package is needed) against the MinIO bucket `docker-compose.yml`
stands up. Images, masks, and embeddings are stored here; `RunState`
carries only the resulting `s3://` URIs (`image_uri`/`expression_uri`),
not raw arrays -- keeping large binary data out of the Postgres checkpoint
rows, per Art. XII §8's split between object storage and Postgres.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import s3fs

from nexus_agent.shared.config import settings

S3_URI_PREFIX = "s3://"


def _fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=settings.s3_access_key,
        secret=settings.s3_secret_key,
        client_kwargs={"endpoint_url": settings.s3_endpoint_url},
    )


def _uri_to_path(uri: str) -> str:
    if not uri.startswith(S3_URI_PREFIX):
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    return uri[len(S3_URI_PREFIX) :]


def _path_to_uri(path: str) -> str:
    return f"{S3_URI_PREFIX}{path}"


def ensure_bucket(bucket: str | None = None) -> None:
    bucket = bucket or settings.s3_bucket
    fs = _fs()
    if not fs.exists(bucket):
        fs.mkdir(bucket)


def put_array(key: str, array: np.ndarray, *, bucket: str | None = None) -> str:
    """Write `array` as .npy bytes under `bucket/key`, returning its s3:// URI."""
    bucket = bucket or settings.s3_bucket
    ensure_bucket(bucket)
    path = f"{bucket}/{key}"
    buffer = io.BytesIO()
    np.save(buffer, array)
    with _fs().open(path, "wb") as f:
        f.write(buffer.getvalue())
    return _path_to_uri(path)


def get_array(uri: str) -> np.ndarray:
    with _fs().open(_uri_to_path(uri), "rb") as f:
        return np.load(f)


def put_json(key: str, obj: Any, *, bucket: str | None = None) -> str:
    bucket = bucket or settings.s3_bucket
    ensure_bucket(bucket)
    path = f"{bucket}/{key}"
    with _fs().open(path, "w") as f:
        json.dump(obj, f)
    return _path_to_uri(path)


def get_json(uri: str) -> Any:
    with _fs().open(_uri_to_path(uri), "r") as f:
        return json.load(f)


def put_anndata(key: str, adata: ad.AnnData, *, bucket: str | None = None) -> str:
    """Write `adata` as `.h5ad` (Art. XII §1's canonical per-sample format)
    under `bucket/key`, returning its s3:// URI.
    """
    bucket = bucket or settings.s3_bucket
    ensure_bucket(bucket)
    path = f"{bucket}/{key}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "data.h5ad"
        adata.write_h5ad(tmp_path)
        with _fs().open(path, "wb") as f:
            f.write(tmp_path.read_bytes())
    return _path_to_uri(path)


def put_bytes(key: str, data: bytes, *, bucket: str | None = None) -> str:
    """Write raw `data` under `bucket/key`, returning its s3:// URI. Used by
    Phase 5's LoRA fine-tune loop (`learning/lora_finetune.py`) to persist
    `torch.save`-serialized checkpoints (adapter + classifier state).
    """
    bucket = bucket or settings.s3_bucket
    ensure_bucket(bucket)
    path = f"{bucket}/{key}"
    with _fs().open(path, "wb") as f:
        f.write(data)
    return _path_to_uri(path)


def get_bytes(uri: str) -> bytes:
    with _fs().open(_uri_to_path(uri), "rb") as f:
        return f.read()


def get_anndata(uri: str) -> ad.AnnData:
    with _fs().open(_uri_to_path(uri), "rb") as f:
        data = f.read()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "data.h5ad"
        tmp_path.write_bytes(data)
        return ad.read_h5ad(tmp_path)
