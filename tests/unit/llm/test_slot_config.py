"""Slot resolution, destination binding, and secret-free provider status tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import AppEnvironment, Settings
from app.llm.config import describe_providers, resolve_slots
from app.schemas.privacy import DispatchPurpose
from app.schemas.providers import ModelSlot, ProviderKind

TEST_SECRET = "stage7-slot-config-test-secret-with-32-chars"
SYNTHETIC_GOOGLE_KEY = "synthetic-google-key-not-real"
SYNTHETIC_NVIDIA_KEY = "synthetic-nvidia-key-not-real"


def make_settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        app_env=AppEnvironment.TEST,
        app_secret_key=TEST_SECRET,
        **overrides,  # type: ignore[arg-type]
    )


def partially_configured_settings() -> Settings:
    return make_settings(
        google_api_key=SYNTHETIC_GOOGLE_KEY,
        nvidia_api_key=SYNTHETIC_NVIDIA_KEY,
        llm_reasoner_a_provider="google",
        llm_reasoner_a_model="synthetic-google-reasoner",
        llm_adjudicator_provider="nvidia",
        llm_adjudicator_model="synthetic-nvidia-adjudicator",
    )


def test_resolve_slots_returns_only_fully_configured_slots() -> None:
    slots = resolve_slots(partially_configured_settings())

    assert set(slots) == {ModelSlot.REASONER_A, ModelSlot.FINAL_ADJUDICATOR}
    assert slots[ModelSlot.REASONER_A].provider is ProviderKind.GOOGLE
    assert slots[ModelSlot.REASONER_A].purpose is DispatchPurpose.REASONING
    assert slots[ModelSlot.FINAL_ADJUDICATOR].provider is ProviderKind.NVIDIA
    assert slots[ModelSlot.FINAL_ADJUDICATOR].purpose is DispatchPurpose.ADJUDICATION


def test_reasoner_slots_are_separate_from_the_adjudicator() -> None:
    settings = make_settings(
        google_api_key=SYNTHETIC_GOOGLE_KEY,
        llm_reasoner_a_provider="google",
        llm_reasoner_a_model="synthetic-a",
        llm_reasoner_b_provider="google",
        llm_reasoner_b_model="synthetic-b",
        llm_reasoner_c_provider="google",
        llm_reasoner_c_model="synthetic-c",
        llm_adjudicator_provider="google",
        llm_adjudicator_model="synthetic-adjudicator",
    )
    slots = resolve_slots(settings)

    assert set(slots) == {
        ModelSlot.REASONER_A,
        ModelSlot.REASONER_B,
        ModelSlot.REASONER_C,
        ModelSlot.FINAL_ADJUDICATOR,
    }


def test_google_slot_endpoint_is_derived_from_the_documented_base_url() -> None:
    slots = resolve_slots(partially_configured_settings())

    destination = slots[ModelSlot.REASONER_A].destination()
    assert destination.endpoint == (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/synthetic-google-reasoner:generateContent"
    )
    assert destination.provider == "google"
    assert destination.model == "synthetic-google-reasoner"


def test_google_models_prefix_is_stripped_from_slot_models() -> None:
    settings = make_settings(
        google_api_key=SYNTHETIC_GOOGLE_KEY,
        llm_reasoner_a_provider="google",
        llm_reasoner_a_model="models/synthetic-google-reasoner",
    )
    slots = resolve_slots(settings)

    assert "models/models/" not in slots[ModelSlot.REASONER_A].endpoint
    assert slots[ModelSlot.REASONER_A].endpoint.endswith(
        "/models/synthetic-google-reasoner:generateContent"
    )


def test_llama_slot_endpoint_combines_base_url_and_route() -> None:
    settings = make_settings(
        llama_base_url="https://llama.example.invalid",
        llama_api_key="synthetic-llama-key-not-real",
        llm_reasoner_a_provider="llama",
        llm_reasoner_a_model="synthetic-llama-model",
    )
    slots = resolve_slots(settings)

    assert (
        slots[ModelSlot.REASONER_A].endpoint
        == "https://llama.example.invalid/v1/chat/completions"
    )


def test_explicit_slot_endpoint_overrides_the_provider_default() -> None:
    settings = make_settings(
        google_api_key=SYNTHETIC_GOOGLE_KEY,
        llm_reasoner_a_provider="google",
        llm_reasoner_a_model="synthetic-google-reasoner",
        llm_reasoner_a_endpoint="https://proxy.example.invalid/v1beta/models/x:generateContent",
    )
    slots = resolve_slots(settings)

    assert (
        slots[ModelSlot.REASONER_A].endpoint
        == "https://proxy.example.invalid/v1beta/models/x:generateContent"
    )


def test_slot_destination_purposes_match_their_roles() -> None:
    settings = make_settings(
        google_api_key=SYNTHETIC_GOOGLE_KEY,
        llm_reasoner_a_provider="google",
        llm_reasoner_a_model="synthetic-a",
        llm_adjudicator_provider="google",
        llm_adjudicator_model="synthetic-adjudicator",
        llm_fallback_provider="google",
        llm_fallback_model="synthetic-fallback",
    )
    slots = resolve_slots(settings)

    assert slots[ModelSlot.REASONER_A].purpose is DispatchPurpose.REASONING
    assert slots[ModelSlot.FINAL_ADJUDICATOR].purpose is DispatchPurpose.ADJUDICATION
    assert slots[ModelSlot.FALLBACK].purpose is DispatchPurpose.FALLBACK


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_reasoner_a_provider", "google"),
        ("llm_reasoner_a_model", "synthetic-a"),
    ],
)
def test_incomplete_slot_configuration_is_rejected(field: str, value: str) -> None:
    kwargs = {"llm_reasoner_a_provider": None, "llm_reasoner_a_model": None}
    kwargs[field] = value

    with pytest.raises(ValidationError):
        make_settings(**kwargs)


def test_nvidia_slot_requires_its_api_key() -> None:
    with pytest.raises(ValidationError):
        make_settings(
            llm_reasoner_a_provider="nvidia",
            llm_reasoner_a_model="synthetic-nvidia-model",
        )


def test_llama_slot_requires_a_base_url() -> None:
    with pytest.raises(ValidationError):
        make_settings(
            llm_reasoner_a_provider="llama",
            llm_reasoner_a_model="synthetic-llama-model",
        )


def test_fallback_enabled_requires_a_configured_fallback_slot() -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_fallback_enabled=True)


def test_fallback_defaults_to_disabled() -> None:
    settings = partially_configured_settings()

    assert settings.llm_fallback_enabled is False
    assert ModelSlot.FALLBACK not in resolve_slots(settings)


def test_describe_providers_reports_status_without_secrets() -> None:
    settings = partially_configured_settings()
    slots = resolve_slots(settings)

    response = describe_providers(settings, slots)

    dump = response.model_dump_json()
    assert SYNTHETIC_GOOGLE_KEY not in dump
    assert SYNTHETIC_NVIDIA_KEY not in dump
    assert "api_key" not in dump.replace("api_key_configured", "")
    for provider in response.providers:
        assert isinstance(provider.api_key_configured, bool)


def test_describe_providers_reflects_configuration_state() -> None:
    response = describe_providers(
        partially_configured_settings(), resolve_slots(partially_configured_settings())
    )

    assert response.external_llm_enabled is False
    assert response.local_only is True
    assert response.fallback_enabled is False
    assert response.fallback_requires_fresh_approval is True
    assert response.prompt_version == "v1"
    by_provider = {provider.provider: provider for provider in response.providers}
    assert by_provider[ProviderKind.GOOGLE].configured is True
    assert by_provider[ProviderKind.GOOGLE].slot_count == 1
    assert by_provider[ProviderKind.NVIDIA].configured is True
    assert by_provider[ProviderKind.LLAMA].configured is False
    by_slot = {slot.slot: slot for slot in response.slots}
    assert by_slot[ModelSlot.REASONER_A].configured is True
    assert by_slot[ModelSlot.REASONER_A].enabled is False  # external LLM disabled
    assert by_slot[ModelSlot.REASONER_B].configured is False


def test_slots_enable_only_when_external_calls_and_not_local_only() -> None:
    settings = partially_configured_settings()
    slots = resolve_slots(settings)

    external_on = settings.model_copy(
        update={"external_llm_enabled": True, "privacy_local_only": False}
    )
    enabled = {
        slot.slot: slot.enabled for slot in describe_providers(external_on, slots).slots
    }
    assert enabled[ModelSlot.REASONER_A] is True
    assert enabled[ModelSlot.FINAL_ADJUDICATOR] is True
    assert enabled[ModelSlot.FALLBACK] is False  # not configured

    local_only = external_on.model_copy(update={"privacy_local_only": True})
    disabled = {
        slot.slot: slot.enabled for slot in describe_providers(local_only, slots).slots
    }
    assert disabled[ModelSlot.REASONER_A] is False
    assert disabled[ModelSlot.FINAL_ADJUDICATOR] is False


def test_configured_endpoints_appear_in_status_but_never_keys() -> None:
    response = describe_providers(
        partially_configured_settings(), resolve_slots(partially_configured_settings())
    )

    adjudicator = next(
        slot for slot in response.slots if slot.slot is ModelSlot.FINAL_ADJUDICATOR
    )
    assert adjudicator.model == "synthetic-nvidia-adjudicator"
    assert adjudicator.endpoint == "https://integrate.api.nvidia.com/v1/chat/completions"
