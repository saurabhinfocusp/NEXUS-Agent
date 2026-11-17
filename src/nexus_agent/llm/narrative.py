"""Report narrative: prose over findings that already passed Critic, with
every sentence checked against those findings by plain code.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from nexus_agent.llm.client import structured_call
from nexus_agent.report.build import _TIER_LABELS, _TIER_STYLES, Report, _tier_for

Tier = Literal["high", "medium", "low"]
_TIER_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


class Finding(BaseModel):
    """All report entries of one claim type sharing one value -- e.g. every
    cell assigned `domain_2`. The narrative cites findings, not individual
    cells: a sample has thousands of per-cell claims but a handful of
    distinct results.
    """

    finding_id: str
    claim_type: str
    value: str
    n_claims: int
    mean_confidence: float
    min_confidence: float
    share_low_confidence: float
    has_citation: bool
    tier: Tier


def summarize_findings(report: Report) -> list[Finding]:
    """Group `report.entries` by `(claim_type, value)`. A finding's tier is
    the tier of its mean confidence. Entries with no value (e.g. a cell
    with neither a type nor a domain) assert nothing and are left out.
    """
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for entry in report.entries:
        if entry.value is not None:
            groups[(entry.claim_type, entry.value)].append(entry)

    findings = []
    for i, ((claim_type, value), entries) in enumerate(sorted(groups.items()), start=1):
        confidences = np.array([e.confidence for e in entries])
        mean = float(confidences.mean())
        findings.append(
            Finding(
                finding_id=f"F{i}",
                claim_type=claim_type,
                value=value,
                n_claims=len(entries),
                mean_confidence=mean,
                min_confidence=float(confidences.min()),
                share_low_confidence=float(np.mean([e.weight_tier == "low" for e in entries])),
                has_citation=any(e.citations for e in entries),
                tier=_tier_for(mean),
            )
        )
    return findings


class NarrativeSentence(BaseModel):
    text: str
    finding_ids: list[str] = Field(description="Ids of the findings this sentence rests on.")


class NarrativeDraft(BaseModel):
    sentences: list[NarrativeSentence]


class VerifiedSentence(BaseModel):
    text: str
    finding_ids: list[str]
    tier: Tier


class RejectedSentence(BaseModel):
    text: str
    reason: str


class Narrative(BaseModel):
    sentences: list[VerifiedSentence]
    rejected: list[RejectedSentence]


def _mentions(text: str, value: str) -> bool:
    """Whether `text` names `value` as a whole term, ignoring case --
    `domain_1` is not mentioned by a sentence about `domain_10`.
    """
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", text, flags=re.IGNORECASE) is not None


def verify_narrative(draft: NarrativeDraft, findings: list[Finding]) -> Narrative:
    """Keep only the sentences the findings support. A sentence is rejected
    if it cites nothing, cites an id that doesn't exist, or names a
    finding's value without citing a finding that has that value (it is
    talking about a result it gave no evidence for). A kept sentence takes
    the tier of its weakest cited finding, so prose can never present a
    result more confidently than the claims behind it (Art. VI §2).

    This checks grounding, not truth: a sentence that cites the right
    finding but mischaracterizes it in words that name no other finding
    passes.
    """
    by_id = {f.finding_id: f for f in findings}
    kept: list[VerifiedSentence] = []
    rejected: list[RejectedSentence] = []

    for sentence in draft.sentences:
        ids = list(dict.fromkeys(sentence.finding_ids))
        if not ids:
            rejected.append(RejectedSentence(text=sentence.text, reason="cites no finding"))
            continue
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            rejected.append(
                RejectedSentence(text=sentence.text, reason=f"cites finding(s) not in the report: {', '.join(unknown)}")
            )
            continue
        cited_values = {by_id[i].value.lower() for i in ids}
        uncited = sorted(
            {f.value for f in findings if f.value.lower() not in cited_values and _mentions(sentence.text, f.value)}
        )
        if uncited:
            rejected.append(
                RejectedSentence(
                    text=sentence.text, reason=f"names result(s) it does not cite a finding for: {', '.join(uncited)}"
                )
            )
            continue
        tier = min((by_id[i].tier for i in ids), key=_TIER_ORDER.__getitem__)
        kept.append(VerifiedSentence(text=sentence.text, finding_ids=ids, tier=tier))

    return Narrative(sentences=kept, rejected=rejected)


_SYSTEM = """You write the summary paragraph of a spatial biology analysis report, for a biologist who will \
read it before the detailed per-claim evidence below it.

You are given the report's findings. Each finding is one result (a spatial domain, a cell type, a niche, an \
enriched pathway) with how many claims support it and how confident the pipeline is in them. These findings \
are the only things you know about this sample.

Write a short summary: what was found, in order of importance, in plain declarative sentences. For every \
sentence, list the ids of the findings it rests on. Each sentence is checked mechanically against the \
findings it cites and is removed if it cites none, or names a result without citing that result's finding \
-- so a sentence that compares two findings must cite both.

State confidence honestly. Where a finding's confidence is low or a large share of its claims are low \
confidence, say so in that sentence instead of smoothing it over. Labels such as "domain_0" or "niche_2" are \
cluster identifiers produced by the pipeline, not anatomical or biological names; refer to them as they are \
and do not guess what tissue structure they correspond to. Do not add biological interpretation, mechanisms, \
or background knowledge that is not in the findings: the reader needs to know what this pipeline measured, \
and anything else would be indistinguishable from a measured result."""


def _findings_prompt(findings: list[Finding]) -> str:
    lines = [
        f"{f.finding_id}: {f.claim_type} = {f.value!r}; {f.n_claims} claim(s); "
        f"mean confidence {f.mean_confidence:.2f} (lowest {f.min_confidence:.2f}); "
        f"{f.share_low_confidence:.0%} of its claims are low confidence; "
        f"{'has' if f.has_citation else 'no'} supporting literature citation"
        for f in findings
    ]
    return "The report's findings:\n\n" + "\n".join(lines)


def write_narrative(report: Report, *, client=None) -> Narrative | None:
    """Draft and verify a narrative for `report`. Returns None when the
    report has no findings to summarize. Raises `llm.client.LLMUnavailable`
    if the model can't be used.
    """
    findings = summarize_findings(report)
    if not findings:
        return None
    draft = structured_call(
        system=_SYSTEM,
        user=_findings_prompt(findings),
        output_format=NarrativeDraft,
        effort="medium",
        client=client,
    )
    return verify_narrative(draft, findings)


def render_narrative_html(narrative: Narrative) -> str:
    """The narrative as an HTML section, each sentence styled by its tier
    with the same `_TIER_STYLES` the per-claim entries use. Says how many
    drafted sentences were removed, so a short summary isn't mistaken for a
    complete one.
    """
    if not narrative.sentences and not narrative.rejected:
        return ""
    items = "\n".join(
        f'<p class="narrative-sentence tier-{s.tier}" style="{_TIER_STYLES[s.tier]}">'
        f"{html.escape(s.text)} "
        f'<span class="narrative-evidence">[{html.escape(_TIER_LABELS[s.tier])}; '
        f"findings: {html.escape(', '.join(s.finding_ids))}]</span></p>"
        for s in narrative.sentences
    )
    removed = (
        f'<p class="narrative-removed">{len(narrative.rejected)} drafted sentence(s) were removed because they '
        "were not supported by the findings they cited.</p>"
        if narrative.rejected
        else ""
    )
    return f"""<section class="narrative">
<h2>Summary</h2>
<p class="narrative-disclosure">Written by a language model from the claims below, which were reviewed first.
Each sentence was checked against the findings it cites; the per-claim evidence below is the record.</p>
{items}
{removed}
</section>"""
