"""Integration test: requires network (one-time ~27MB DSB2018 dataset fetch,
cached afterward in <repo>/data/benchmark/) and CellPose model weights.

Confirms Phase 3's exit criterion: a real AJI measurement recorded on real
(if not one of Art. VIII §1's four named platforms -- see
`benchmark/aji.py`'s module docstring) annotated data.
"""

from pathlib import Path

import pytest

from nexus_agent.benchmark.aji import run_aji_benchmark

pytestmark = pytest.mark.integration

BENCHMARK_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark"


def test_aji_benchmark_measures_a_real_score():
    result = run_aji_benchmark(BENCHMARK_DATA_DIR, n_images=3)

    assert result["n_images"] == 3
    assert 0.0 <= result["mean_aji"] <= 1.0
    assert len(result["per_image"]) == 3
    for entry in result["per_image"]:
        assert 0.0 <= entry["aji"] <= 1.0

    print(f"\nDSB2018 demo AJI (n={result['n_images']}): mean={result['mean_aji']:.3f}")
    for entry in result["per_image"]:
        print(f"  {entry['image']}: aji={entry['aji']:.3f} n_pred_cells={entry['n_pred_cells']}")
