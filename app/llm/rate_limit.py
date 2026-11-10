"""Small per-provider request limiter shared by every slot using that provider."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class AsyncRateLimiter:
    """Space request starts by a fixed interval without unbounded queueing."""

    def __init__(
        self,
        per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self._interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Reserve the next request slot, waiting outside the lock when needed."""

        async with self._lock:
            now = self._clock()
            wait_seconds = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if wait_seconds > 0:
            await self._sleep(wait_seconds)
