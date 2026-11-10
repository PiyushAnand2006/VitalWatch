"""Versioned prompt construction and normalized request contract tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.llm.prompts import PROMPT_VERSION, PROMPTS, PromptKind, build_llm_request
from app.schemas.providers import LLMMessage, LLMRequest
from tests.unit.llm.conftest import make_payload

EXPECTED_JSON_SHAPE = {
    PromptKind.REASONING: '"facts"',
    PromptKind.CRITIQUE: '"unsupported_claims"',
    PromptKind.REVISION: "bounded revision",
    PromptKind.ADJUDICATION: '"verified_facts"',
}


@pytest.mark.parametrize("kind", list(PromptKind))
def test_every_prompt_kind_builds_a_normalized_request(kind: PromptKind) -> None:
    payload = make_payload()

    request = build_llm_request(
        kind=kind,
        payload=payload,
        request_id="req-prompt-001",
        max_output_tokens=1024,
        temperature=0.1,
    )

    assert request.prompt_id == kind.value
    assert request.prompt_version == PROMPT_VERSION
    assert request.payload is payload
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.max_output_tokens == 1024
    assert request.temperature == 0.1
    # The approved payload is embedded verbatim; nothing new is invented.
    serialized = json.dumps(payload.model_dump(mode="json"), sort_keys=True)
    assert serialized in request.messages[1].content


def test_prompt_roles_have_distinct_instructions_and_one_version() -> None:
    assert len({definition.version for definition in PROMPTS.values()}) == 1
    assert PROMPT_VERSION == "v1"
    for kind, marker in EXPECTED_JSON_SHAPE.items():
        definition = PROMPTS[kind]
        assert marker in definition.system
        assert definition.system.startswith("You are a clinical evidence explainer")
    assert (
        PROMPTS[PromptKind.REASONING].system != PROMPTS[PromptKind.ADJUDICATION].system
    )


def test_adjudication_prompt_forbids_reintroducing_rejected_claims() -> None:
    system = PROMPTS[PromptKind.ADJUDICATION].system
    assert "Never reintroduce a rejected claim" in system
    assert "majority vote" in system


def test_prompts_treat_payload_content_as_untrusted_data() -> None:
    system = PROMPTS[PromptKind.REASONING].system
    assert "untrusted data" in system
    assert "Never invent values" in system
    assert "do not" in system.lower() or "never" in system.lower()


def test_request_rejects_peer_style_message_orders() -> None:
    payload = make_payload()
    base = {
        "request_id": "req-prompt-002",
        "prompt_id": "reasoning",
        "prompt_version": "v1",
        "payload": payload,
        "max_output_tokens": 64,
        "temperature": 0.1,
    }

    with pytest.raises(ValidationError):
        LLMRequest(
            **base,
            messages=(
                LLMMessage(role="user", content="Explain."),
                LLMMessage(role="system", content="Rules."),
            ),
        )
    with pytest.raises(ValidationError):
        LLMRequest(
            **base,
            messages=(
                LLMMessage(role="system", content="Rules."),
                LLMMessage(role="system", content="More rules."),
            ),
        )
    with pytest.raises(ValidationError):
        LLMRequest(**base, messages=(LLMMessage(role="system", content="Only one."),))


def test_request_is_frozen_and_cannot_gain_extra_fields() -> None:
    payload = make_payload()
    request = build_llm_request(
        kind=PromptKind.REASONING,
        payload=payload,
        request_id="req-prompt-003",
        max_output_tokens=64,
        temperature=0.1,
    )

    with pytest.raises(ValidationError):
        request.temperature = 1.5  # type: ignore[misc]
    with pytest.raises(ValidationError):
        LLMRequest(
            **{
                **request.model_dump(),
                "messages": list(request.messages),
                "attacker_field": "nope",
            }
        )
