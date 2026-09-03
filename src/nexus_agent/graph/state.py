"""LangGraph run state (Constitution Art. IV §1).

`history` accumulates every MessageEnvelope emitted by any node via the
`operator.add` reducer, so the full history of a task is inspectable from
the checkpointer at any point, not only the final result.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict
from uuid import UUID

from nexus_agent.shared.schemas import AgentName, AnalyticalGoal, MessageEnvelope, Verdict


class RunState(TypedDict, total=False):
    run_id: UUID
    task_id: UUID
    trace_id: UUID
    goal: AnalyticalGoal
    subtask_plan: list[AgentName]
    # Object-store references (Art. XII §8), not inline arrays -- keeps
    # large binary data out of the Postgres checkpoint rows. Set to route
    # Vision/Analyst through their real model stacks (Phase 3); when unset,
    # they fall back to the lightweight synthetic-cell stub from Phase 0/1
    # so fast/unit tests stay Docker- and model-free.
    image_uri: str | None
    expression_uri: str | None
    history: Annotated[list[MessageEnvelope], operator.add]
    verdict: Verdict | None
    # Test-only hook: lets a test force Critic's stub verdict to exercise the
    # veto/escalate routing without real quality-control logic (Phase 1+).
    # Consumed (cleared) by critic_node after one use to avoid infinite loops.
    force_verdict: Verdict | None
    # Test-only hook: lets a test make a specialist agent's *first* attempt
    # genuinely low-confidence, so Critic's real threshold logic (not
    # force_verdict) triggers the veto (Art. IV §3). Unlike force_verdict,
    # this is consumed implicitly -- a node ignores its own entry here once
    # it detects it's retrying after a Critic veto (see agents/common.py's
    # is_retry_after_veto), so the retry isn't artificially suppressed again.
    stub_confidence_override: dict[AgentName, float] | None
    # Phase 4 (Art. VI §2-3): the confidence-weighted report `_report_node`
    # renders on the `pass` path, once real Analyst claims + XAI evidence
    # bundles exist. `None` on the stub path (unchanged Phase 0-3 behavior)
    # and whenever there are no claims to report on.
    report_html: str | None
