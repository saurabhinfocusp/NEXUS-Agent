import pytest
from pydantic import ValidationError

from nexus_agent.agents.analyst import AnalystClaim
from nexus_agent.agents.biology import EnrichmentClaim
from nexus_agent.agents.qc import QCReport
from nexus_agent.agents.spatial import SpatialClaim
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


def test_qc_report_requires_a_verdict():
    with pytest.raises(ValidationError):
        QCReport()


def test_qc_report_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        QCReport(qc_verdict="unknown")


def test_qc_report_accepts_minimal_valid_shape():
    report = QCReport(qc_verdict="pass")
    assert report.total_counts_mean is None
    assert report.qc_verdict == "pass"


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01])
def test_spatial_claim_confidence_bounds(bad_confidence):
    with pytest.raises(ValidationError):
        SpatialClaim(cell_id="c0", niche_label="niche_0", supporting_stat={}, confidence=bad_confidence)


def test_spatial_claim_requires_niche_label():
    with pytest.raises(ValidationError):
        SpatialClaim(cell_id="c0", supporting_stat={}, confidence=0.5)


def test_spatial_claim_accepts_valid_shape():
    claim = SpatialClaim(cell_id="c0", niche_label="niche_0", supporting_stat={"stat": 1.0}, confidence=0.5)
    assert claim.niche_label == "niche_0"


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01])
def test_enrichment_claim_confidence_bounds(bad_confidence):
    with pytest.raises(ValidationError):
        EnrichmentClaim(
            cell_id="pathway-0",
            pathway="pathway-0",
            gene_set=["G1"],
            p_value=0.01,
            source_db="Enrichr:KEGG_2021_Human",
            confidence=bad_confidence,
        )


@pytest.mark.parametrize("bad_p", [-0.01, 1.01])
def test_enrichment_claim_p_value_bounds(bad_p):
    with pytest.raises(ValidationError):
        EnrichmentClaim(
            cell_id="pathway-0",
            pathway="pathway-0",
            gene_set=["G1"],
            p_value=bad_p,
            source_db="Enrichr:KEGG_2021_Human",
            confidence=0.5,
        )


def test_enrichment_claim_accepts_valid_shape():
    claim = EnrichmentClaim(
        cell_id="pathway-0",
        pathway="pathway-0",
        gene_set=["G1", "G2"],
        p_value=0.01,
        fdr=0.02,
        source_db="Enrichr:KEGG_2021_Human",
        confidence=0.9,
    )
    assert claim.pathway == "pathway-0"
    assert claim.gene_set == ["G1", "G2"]
