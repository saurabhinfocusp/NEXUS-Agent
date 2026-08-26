"""Assembles the Coordinator -> Vision -> Analyst -> Critic StateGraph
(Constitution Art. III, Art. IV, Art. XII §4).

Critic's verdict is a conditional edge: `pass` -> report node, `veto` ->
back to the originating agent's node with the objection appended to
payload, `escalate` -> human-review node. No edge exists from Vision or
Analyst directly to a terminal node -- everything routes through Critic
(Art. III §2).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nexus_agent.graph.nodes import (
    analyst_node,
    coordinator_node,
    critic_node,
    human_review_node,
    report_node,
    vision_node,
)
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import Verdict


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


def build_graph() -> StateGraph:
    graph = StateGraph(RunState)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("vision", vision_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("report", report_node)
    graph.add_node("human_review", human_review_node)

    graph.set_entry_point("coordinator")
    graph.add_edge("coordinator", "vision")
    graph.add_edge("vision", "analyst")
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
