"""Natural-language intake: a question in, an `AnalyticalGoal` out."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_agent.llm.client import structured_call
from nexus_agent.shared.schemas import AnalyticalGoal, Modality

_SYSTEM = """You configure a spatial biology analysis pipeline for a researcher's question.

The pipeline always does these, with no choice to make: data quality control, cell segmentation and \
morphology embedding from the histology image, fusion of morphology with gene expression, spatial-domain \
assignment per cell, and cell-type calls where a trained classifier exists.

Two further analyses are optional, and you decide whether the question calls for each:
- Spatial analysis: neighbourhood enrichment between cell groups, spatial co-occurrence, and spatially \
variable genes (Moran's I). Turn it on when the question is about how cells are arranged, which groups sit \
next to each other, niches, or spatial gene-expression patterns.
- Enrichment analysis: pathway and gene-set enrichment (Enrichr, g:Profiler) and protein-interaction support \
(STRING) for the sample's most variable genes. Turn it on when the question is about pathways, biological \
processes, or mechanisms. It queries external services, so leave it off when the question doesn't need it.

The pipeline cannot do anything else. If part of the question needs something outside this list -- \
comparing samples, differential expression between conditions, trajectory inference, survival or clinical \
outcome analysis, and so on -- list that part under out_of_scope in the researcher's own terms, so they are \
told plainly instead of receiving a report that silently ignores it."""


class GoalPlan(BaseModel):
    run_spatial_analysis: bool
    run_enrichment_analysis: bool
    rationale: str = Field(description="One or two sentences on why each optional analysis is on or off.")
    out_of_scope: list[str] = Field(description="Parts of the question this pipeline cannot answer; empty if none.")


def plan_goal(
    question: str, *, sample_id: str, modalities: list[Modality], client=None
) -> tuple[AnalyticalGoal, GoalPlan]:
    """Choose the optional analyses for `question`.

    `modalities` is what data was actually supplied and is never the
    model's to change. Raises `llm.client.LLMUnavailable` if the model
    can't be used; the caller should fall back to its default goal.
    """
    plan = structured_call(
        system=_SYSTEM,
        user=f"Data supplied for this sample: {', '.join(modalities)}.\n\nThe researcher's question:\n{question}",
        output_format=GoalPlan,
        effort="low",
        client=client,
    )
    goal = AnalyticalGoal(
        sample_id=sample_id,
        modalities=modalities,
        # Both analyses read expression data; without it the flags are inert
        # (Coordinator ignores them), so don't record them as requested.
        run_spatial_analysis=plan.run_spatial_analysis and "expression" in modalities,
        run_enrichment_analysis=plan.run_enrichment_analysis and "expression" in modalities,
    )
    return goal, plan
