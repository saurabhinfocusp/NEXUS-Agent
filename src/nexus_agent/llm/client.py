"""Shared plumbing for the optional LLM layer."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from nexus_agent.shared.config import settings

T = TypeVar("T", bound=BaseModel)

# Lets the API re-run a request on Anthropic's recommended fallback model
# when a safety classifier declines it, instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    """The LLM layer can't be used: package missing, no credentials, the
    request failed, or the model declined. Callers treat this as "carry on
    without the LLM", never as a failed run.
    """


def get_client():
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable('the anthropic package is not installed; run `pip install -e ".[llm]"`') from exc
    return anthropic.Anthropic()


def structured_call(*, system: str, user: str, output_format: type[T], effort: str, client=None) -> T:
    """One request returning a validated `output_format` instance."""
    import anthropic

    client = client or get_client()
    try:
        response = client.beta.messages.parse(
            model=settings.llm_model,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": effort},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
        )
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable(f"could not reach the Anthropic API: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.AnthropicError as exc:  # e.g. no credentials configured
        raise LLMUnavailable(str(exc)) from exc

    if response.stop_reason == "refusal":
        raise LLMUnavailable("the model declined this request")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise LLMUnavailable(f"no complete structured output (stop_reason={response.stop_reason})")
    return response.parsed_output
