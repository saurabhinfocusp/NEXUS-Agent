"""Assembles the Coordinator -> Vision -> Analyst -> Critic StateGraph
(Constitution Art. III, Art. IV, Art. XII §4).

Two real routing decisions: Coordinator's conditional edge sends the task
to whichever specialist its subtask plan starts with (Art. III §1), and
Vision's conditional edge sends it on to Analyst only if Analyst is also in
the plan, else straight to Critic. Critic's verdict is a conditional edge:
`pass` -> report node, `veto` -> back to the originating agent's node with
the objection appended to payload, `escalate` -> human-review node. No edge
exists from Vision or Analyst directly to a terminal node -- everything
routes through Critic (Art. III §2).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nexus_agent.agents.analyst import analyst_node
from nexus_agent.agents.coordinator import coordinator_node
from nexus_agent.agents.critic import critic_node
from nexus_agent.agents.vision import vision_node
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, Verdict


def _report_node(state: RunState) -> dict:
    """Real report assembly (Phase 4, Art. VI §2-3) once Analyst produced real
    claims; on the Phase 0/1 stub path (no claims, or no evidence bundles
    stored by Critic) this stays a no-op, unchanged from before Phase 4.
    """
    if not (state.get("image_uri") or state.get("expression_uri")):
        return {}  # Phase 0/1 stub path: unchanged, no-op (no real infra touched)

    analyst_message = next((m for m in state["history"] if m.from_agent == AgentName.ANALYST), None)
    claims_raw = analyst_message.payload.get("claims", []) if analyst_message else []
    if not claims_raw:
        return {}

    from nexus_agent.report.build import ReportClaimInput, render_report, render_report_html
    from nexus_agent.shared.config import settings
    from nexus_agent.xai.claims_evidence import fetch_evidence_bundle

    report_claims = [
        ReportClaimInput(
            claim_id=c["cell_id"],
            cell_type=c.get("cell_type"),
            spatial_domain=c.get("spatial_domain"),
            confidence=c["confidence"],
        )
        for c in claims_raw
    ]

    evidence_by_claim_id: dict[str, dict] = {}
    for claim in report_claims:
        try:
            bundle = fetch_evidence_bundle(settings.postgres_dsn, state["run_id"], claim.claim_id)
        except Exception:
            bundle = None
        if bundle is not None:
            evidence_by_claim_id[claim.claim_id] = {
                "heatmap_uri": bundle.heatmap_uri,
                "shap_top_genes": bundle.shap_top_genes,
                "citations": bundle.citations,
                "no_literature_retrieved": bundle.no_literature_retrieved,
            }

    report = render_report(state["run_id"], state["task_id"], report_claims, evidence_by_claim_id)
    return {"report_html": render_report_html(report)}


def _human_review_node(state: RunState) -> dict:
    """Terminal node for the `escalate` path. Phase 5 (Art. VII §3) gives it
    an actual persisted queue via `review/escalation.py`, gated to the real
    pipeline (image_uri/expression_uri set) so the fast stub-path tests
    (force_verdict=ESCALATE in tests/test_critic.py, tests/test_graph_smoke.py,
    tests/test_error_recovery.py) never attempt a Postgres connection --
    unchanged from before Phase 5.
    """
    if not (state.get("image_uri") or state.get("expression_uri")):
        return {}

    escalated_message = state["history"][-2]  # the message Critic escalated (Art. IV §3)
    try:
        from nexus_agent.review.escalation import record_escalation
        from nexus_agent.shared.config import settings

        record_escalation(
            settings.postgres_dsn,
            run_id=state["run_id"],
            task_id=state["task_id"],
            claim_id=None,
            payload=escalated_message.payload,
            confidence=escalated_message.confidence,
        )
    except Exception:
        pass  # best-effort persistence; never block graph completion on it
    return {}


def _coordinator_router(state: RunState) -> str:
    return state["subtask_plan"][0].value


def _vision_router(state: RunState) -> str:
    return "analyst" if AgentName.ANALYST in state["subtask_plan"] else "critic"


def _critic_router(state: RunState) -> str:
    verdict = state.get("verdict")
    if verdict == Verdict.PASS:
        return "report"
    if verdict == Verdict.ESCALATE:
        return "human_review"
    if verdict == Verdict.VETO:
        # The message critic just reviewed (second-to-last, since critic's own
        # envelope was appended last) tells us who to route back to.
        return state["history"][-2].from_agent.value
    raise ValueError(f"critic produced no routable verdict: {verdict!r}")


def _with_message_logging(node_fn):
    """Wrap a node so every `MessageEnvelope` it emits is also persisted into
    `message_log` (Art. XII §8: veto rate/latency/confidence-distribution
    dashboards must be measured from a real queryable store, not mined from
    checkpointer blobs). Gated to the real pipeline (image_uri/expression_uri
    set) so the fast stub-path tests never attempt a Postgres connection --
    unchanged from before Phase 5.
    """

    def wrapped(state: RunState) -> dict:
        result = node_fn(state)
        messages = result.get("history") if isinstance(result, dict) else None
        if messages and (state.get("image_uri") or state.get("expression_uri")):
            try:
                import psycopg

                from nexus_agent.shared.config import settings

                with psycopg.connect(settings.postgres_dsn, autocommit=True) as conn:
                    for message in messages:
                        conn.execute(
                            """
                            INSERT INTO message_log
                                (run_id, task_id, from_agent, to_agent, verdict, confidence)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                message.run_id,
                                message.task_id,
                                message.from_agent.value,
                                message.to_agent.value,
                                message.payload.get("verdict"),
                                message.confidence,
                            ),
                        )
            except Exception:
                pass  # best-effort dashboard substrate; never block the graph on it
        return result

    return wrapped


def build_graph() -> StateGraph:
    graph = StateGraph(RunState)
    graph.add_node("coordinator", _with_message_logging(coordinator_node))
    graph.add_node("vision", _with_message_logging(vision_node))
    graph.add_node("analyst", _with_message_logging(analyst_node))
    graph.add_node("critic", _with_message_logging(critic_node))
    graph.add_node("report", _report_node)
    graph.add_node("human_review", _human_review_node)

    graph.set_entry_point("coordinator")
    graph.add_conditional_edges("coordinator", _coordinator_router, {"vision": "vision", "analyst": "analyst"})
    graph.add_conditional_edges("vision", _vision_router, {"analyst": "analyst", "critic": "critic"})
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges(
        "critic",
        _critic_router,
        {"report": "report", "human_review": "human_review", "vision": "vision", "analyst": "analyst"},
    )
    graph.add_edge("report", END)
    graph.add_edge("human_review", END)
    return graph


def compile_with_memory() -> CompiledStateGraph:
    """Fast in-process compile for unit tests -- no Postgres required."""
    return build_graph().compile(checkpointer=MemorySaver())


@contextmanager
def compile_with_postgres(conn_string: str) -> Iterator[CompiledStateGraph]:
    """Compile against a real PostgresSaver. Runs `setup()` once per call;
    cheap/idempotent, fine for dev and test use.
    """
    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        checkpointer.setup()
        yield build_graph().compile(checkpointer=checkpointer)
