import uuid

import pytest
from pydantic import ValidationError

from nexus_agent.shared.schemas import AgentName, MessageEnvelope


def _base_kwargs(**overrides):
    kwargs = dict(
        run_id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        from_agent=AgentName.COORDINATOR,
        to_agent=AgentName.VISION,
        payload={"note": "hello"},
        confidence=0.9,
        trace_id=uuid.uuid4(),
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_envelope_round_trips_through_json():
    envelope = MessageEnvelope(**_base_kwargs())
    restored = MessageEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored == envelope


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, -1.0, 2.0])
def test_confidence_out_of_bounds_rejected(bad_confidence):
    with pytest.raises(ValidationError):
        MessageEnvelope(**_base_kwargs(confidence=bad_confidence))


def test_confidence_is_required():
    kwargs = _base_kwargs()
    del kwargs["confidence"]
    with pytest.raises(ValidationError):
        MessageEnvelope(**kwargs)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        MessageEnvelope(**_base_kwargs(), unexpected_field="nope")


def test_invalid_agent_name_rejected():
    with pytest.raises(ValidationError):
        MessageEnvelope(**_base_kwargs(from_agent="not-a-real-agent"))
