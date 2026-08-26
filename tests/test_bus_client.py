"""Unit tests for the bus's validate-before-publish behavior.

Uses a stub in place of the real redis.Redis client so this stays a fast,
Docker-free test -- the integration-level "does it actually talk to Redis"
check belongs with test_graph_postgres.py-style infra tests, not here.
"""

import uuid

import pytest

from nexus_agent.bus.client import EnvelopeRejected, MessageBus
from nexus_agent.shared.schemas import AgentName, MessageEnvelope


class _StubRedis:
    def __init__(self):
        self.added: list[tuple[str, dict]] = []

    def xadd(self, stream, fields):
        self.added.append((stream, fields))
        return f"stub-id-{len(self.added)}"


def _valid_envelope() -> MessageEnvelope:
    return MessageEnvelope(
        run_id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        from_agent=AgentName.COORDINATOR,
        to_agent=AgentName.VISION,
        payload={},
        confidence=0.75,
        trace_id=uuid.uuid4(),
    )


def test_valid_envelope_is_published():
    stub = _StubRedis()
    bus = MessageBus(client=stub)

    message_id = bus.publish("nexus.events", _valid_envelope())

    assert message_id == "stub-id-1"
    assert len(stub.added) == 1


def test_invalid_message_is_rejected_not_forwarded():
    stub = _StubRedis()
    bus = MessageBus(client=stub)

    with pytest.raises(EnvelopeRejected):
        bus.publish("nexus.events", {"confidence": 1.5})  # missing required fields, bad confidence

    assert stub.added == []
