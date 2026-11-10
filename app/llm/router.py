"""Slot routing with mandatory, per-destination approval before every provider call."""

from __future__ import annotations

import logging
from uuid import UUID

from app.llm.base import LLMProvider
from app.llm.config import SlotConfig
from app.llm.errors import LLMProviderError
from app.privacy.dispatch import ApprovedDispatcher
from app.schemas.privacy import (
    DispatchOutcome,
    DispatchPurpose,
    ExternalDestination,
    PreparedPayload,
)
from app.schemas.providers import (
    LLMErrorCode,
    LLMRequest,
    LLMResponse,
    ModelSlot,
    RoutingResult,
    RoutingStatus,
)

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Route a logical slot only after its exact destination is approved."""

    def __init__(
        self,
        *,
        slots: dict[ModelSlot, SlotConfig],
        providers: dict[ModelSlot, LLMProvider],
        dispatcher: ApprovedDispatcher,
        external_enabled: bool,
        local_only: bool,
        fallback_enabled: bool,
    ) -> None:
        self._slots = slots
        self._providers = providers
        self._dispatcher = dispatcher
        self._external_enabled = external_enabled
        self._local_only = local_only
        self._fallback_enabled = fallback_enabled

    async def generate(
        self,
        *,
        slot: ModelSlot,
        request: LLMRequest,
        approval_id: UUID,
        fallback_approval_id: UUID | None = None,
        purpose: DispatchPurpose | None = None,
    ) -> RoutingResult:
        """Call one slot, optionally using a separately approved fallback route.

        ``purpose`` overrides the slot's default destination purpose for debate
        calls (critique, revision, repair); each purpose is a distinct approval.
        """

        unavailable = self._unavailable_result(slot)
        if unavailable is not None:
            return unavailable

        config = self._slots[slot]
        provider = self._providers[slot]
        destination = config.destination(purpose)
        primary = await self._dispatch(
            approval_id=approval_id,
            request=request,
            provider=provider,
            slot=slot,
            destination=destination,
        )
        if primary.status is not RoutingStatus.FAILED:
            return primary

        error = primary.error_code
        if (
            not self._fallback_enabled
            or fallback_approval_id is None
            or error is None
            or error
            in {
                LLMErrorCode.AUTHENTICATION_FAILED,
                LLMErrorCode.BAD_REQUEST,
                LLMErrorCode.MALFORMED_RESPONSE,
                LLMErrorCode.EMPTY_RESPONSE,
                LLMErrorCode.BLOCKED_RESPONSE,
            }
        ):
            return primary

        fallback_unavailable = self._unavailable_result(ModelSlot.FALLBACK)
        if fallback_unavailable is not None:
            return primary
        fallback_config = self._slots[ModelSlot.FALLBACK]
        fallback_provider = self._providers[ModelSlot.FALLBACK]
        fallback = await self._dispatch(
            approval_id=fallback_approval_id,
            request=request,
            provider=fallback_provider,
            slot=ModelSlot.FALLBACK,
            destination=fallback_config.destination(),
        )
        return fallback.model_copy(update={"fallback_used": True})

    def _unavailable_result(self, slot: ModelSlot) -> RoutingResult | None:
        if not self._external_enabled or self._local_only:
            return RoutingResult(
                status=RoutingStatus.DISABLED,
                slot=slot,
                error_code=LLMErrorCode.EXTERNAL_DISABLED,
            )
        if slot not in self._slots or slot not in self._providers:
            return RoutingResult(
                status=RoutingStatus.DISABLED,
                slot=slot,
                error_code=LLMErrorCode.SLOT_DISABLED,
            )
        if slot is ModelSlot.FALLBACK and not self._fallback_enabled:
            return RoutingResult(
                status=RoutingStatus.DISABLED,
                slot=slot,
                error_code=LLMErrorCode.SLOT_DISABLED,
            )
        return None

    async def _dispatch(
        self,
        *,
        approval_id: UUID,
        request: LLMRequest,
        provider: LLMProvider,
        slot: ModelSlot,
        destination: ExternalDestination,
    ) -> RoutingResult:
        try:
            response = await self._dispatcher.dispatch(
                approval_id=approval_id,
                payload=request.payload,
                destination=destination,
                transport=lambda payload, bound: self._transport(payload, bound, provider, request),
            )
        except LLMProviderError as exc:
            logger.warning(
                "external model call failed provider=%s model=%s code=%s",
                provider.kind.value,
                provider.model,
                exc.code.value,
            )
            return RoutingResult(
                status=RoutingStatus.FAILED,
                slot=slot,
                destination=destination,
                dispatch_outcome=DispatchOutcome.DISPATCHED,
                error_code=exc.code,
            )
        if response.outcome is not DispatchOutcome.DISPATCHED:
            return RoutingResult(
                status=RoutingStatus.BLOCKED,
                slot=slot,
                destination=destination,
                dispatch_outcome=response.outcome,
                error_code=LLMErrorCode.APPROVAL_REQUIRED,
            )
        value = response.value
        if not isinstance(value, LLMResponse):
            return RoutingResult(
                status=RoutingStatus.FAILED,
                slot=slot,
                destination=destination,
                dispatch_outcome=response.outcome,
                error_code=LLMErrorCode.MALFORMED_RESPONSE,
            )
        return RoutingResult(
            status=RoutingStatus.COMPLETED,
            slot=slot,
            destination=destination,
            dispatch_outcome=response.outcome,
            response=value,
        )

    async def _transport(
        self,
        payload: PreparedPayload,
        destination: ExternalDestination,
        provider: LLMProvider,
        request: LLMRequest,
    ) -> LLMResponse:
        del destination
        if payload != request.payload:
            raise LLMProviderError(LLMErrorCode.APPROVAL_REQUIRED)
        return await provider.generate(request)
