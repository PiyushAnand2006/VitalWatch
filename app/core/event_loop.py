"""Cross-platform event loop compatibility for the async PostgreSQL driver."""

import asyncio
import sys


def ensure_windows_selector_event_loop() -> None:
    """Select the loop psycopg's async mode supports on Windows (no-op elsewhere)."""

    if sys.platform != "win32":
        return
    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is not None:
        asyncio.set_event_loop_policy(policy())
