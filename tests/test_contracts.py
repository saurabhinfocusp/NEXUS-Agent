import pytest
from pydantic import ValidationError

from nexus_agent.agents.analyst import AnalystClaim
from nexus_agent.agents.vision import EMBEDDING_DIM, VisionCellRecord
from nexus_agent.shared.schemas import AnalyticalGoal


def test_analytical_goal_requires_at_least_one_modality():
    with pytest.raises(ValidationError):
        AnalyticalGoal(sample_id="s1", modalities=[])


def test_analytical_goal_rejects_unknown_modality():
    with pytest.raises(ValidationError):
        AnalyticalGoal(sample_id="s1", modalities=["proteomics"])


def test_analytical_goal_accepts_valid_modalities():
    goal = AnalyticalGoal(sample_id="s1", modalities=["image", "expression"])
    assert goal.modalities == ["image", "expression"]


def test_vision_cell_record_requires_exact_embedding_dim():
    kwargs = dict(
        cell_id="c0",
        centroid_xy=(1.0, 2.0),
        mask_polygon=[(0.0, 0.0), (1.0, 1.0)],
        embedding_model_version="v1",
    )
    VisionCellRecord(**kwargs, embedding_vector=[0.0] * EMBEDDING_DIM)  # exact length OK

    with pytest.raises(ValidationError):
        VisionCellRecord(**kwargs, embedding_vector=[0.0] * (EMBEDDING_DIM - 1))
    with pytest.raises(ValidationError):
        VisionCellRecord(**kwargs, embedding_vector=[0.0] * (EMBEDDING_DIM + 1))


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01])
def test_analyst_claim_confidence_bounds(bad_confidence):
    with pytest.raises(ValidationError):
        AnalystClaim(cell_id="c0", confidence=bad_confidence, provisional=False)


def test_analyst_claim_requires_provisional_flag():
    with pytest.raises(ValidationError):
        AnalystClaim(cell_id="c0", confidence=0.5)
