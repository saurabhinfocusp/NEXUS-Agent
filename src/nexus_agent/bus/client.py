"""Redis Streams message bus (Constitution Art. XII §4).

"A message failing schema validation is rejected at the bus, not
forwarded — this is the enforcement mechanism behind Article III, Section
3." Concretely: `publish()` validates against MessageEnvelope before
XADD; a validation failure raises and nothing reaches the stream.
"""

from __future__ import annotations

import uuid

import redis
from pydantic import ValidationError

from nexus_agent.shared.config import settings
from nexus_agent.shared.schemas import MessageEnvelope

_ENVELOPE_FIELD = "envelope"


class EnvelopeRejected(Exception):
    """Raised when a message fails MessageEnvelope validation and is not published."""


def get_redis_client() -> redis.Redis:
    return redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)


class MessageBus:
    def __init__(self, client: redis.Redis | None = None) -> None:
        self._client = client or get_redis_client()

    def publish(self, stream: str, message: MessageEnvelope | dict) -> str:
        """Validate `message` as a MessageEnvelope and XADD it to `stream`.

        Raises EnvelopeRejected (never reaches the stream) if validation fails.
        """
        try:
            envelope = (
                message if isinstance(message, MessageEnvelope) else MessageEnvelope.model_validate(message)
            )
        except ValidationError as exc:
            raise EnvelopeRejected(str(exc)) from exc

        return self._client.xadd(stream, {_ENVELOPE_FIELD: envelope.model_dump_json()})

    def ensure_group(self, stream: str, group: str) -> None:
        try:
            self._client.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, MessageEnvelope]]:
        """Read up to `count` pending messages for `consumer` in `group`."""
        self.ensure_group(stream, group)
        response = self._client.xreadgroup(group, consumer, {stream: ">"}, count=count, block=block_ms)
        results: list[tuple[str, MessageEnvelope]] = []
        for _stream_name, entries in response or []:
            for message_id, fields in entries:
                results.append((message_id, MessageEnvelope.model_validate_json(fields[_ENVELOPE_FIELD])))
        return results

    def ack(self, stream: str, group: str, message_id: str) -> None:
        self._client.xack(stream, group, message_id)


def new_trace_id() -> uuid.UUID:
    return uuid.uuid4()
