"""Integration test: requires `docker compose up -d` (MinIO reachable)."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from nexus_agent.data import object_store

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_minio():
    try:
        object_store.ensure_bucket()
    except Exception as exc:  # noqa: BLE001 -- any connection failure means "not reachable"
        pytest.skip(f"MinIO not reachable ({exc}); run `docker compose up -d`.")


def test_array_round_trips_through_object_store():
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    uri = object_store.put_array("tests/array.npy", array)
    assert uri.startswith("s3://")
    assert np.array_equal(object_store.get_array(uri), array)


def test_json_round_trips_through_object_store():
    payload = {"a": 1, "b": [1, 2, 3]}
    uri = object_store.put_json("tests/meta.json", payload)
    assert object_store.get_json(uri) == payload


def test_anndata_round_trips_through_object_store():
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.random((5, 3)).astype(np.float32), obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]))
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(5, 2))

    uri = object_store.put_anndata("tests/sample.h5ad", adata)
    restored = object_store.get_anndata(uri)

    assert np.allclose(restored.X, adata.X)
    assert np.allclose(restored.obsm["spatial"], adata.obsm["spatial"])
