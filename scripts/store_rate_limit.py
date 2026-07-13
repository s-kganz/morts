"""Rate-limited wrapper for async object-store range reads."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from async_geotiff import Store

SleepCallable = Callable[[float], Awaitable[object]]
RangeBatchCost = Callable[
    [str, Sequence[int], Sequence[int] | None, Sequence[int] | None],
    float,
]

logger = logging.getLogger(__name__)


class ObjectStoreRateLimiter:
    """Loop-agnostic token-bucket limiter for async object-store operations.

    The limiter uses a small ``threading.Lock``-protected reservation clock
    instead of asyncio locks or events, so one instance can be shared by stores
    used from different event loops and threads. Waiting happens with
    ``asyncio.sleep`` after the reservation is made, which avoids blocking the
    active event loop.
    """

    def __init__(
        self,
        rate: float,
        *,
        per: float = 1.0,
        burst: float = 1.0,
    ) -> None:
        """Create a limiter allowing ``rate`` operations per ``per`` seconds.

        Args:
            rate: Number of operation tokens replenished per period.
            per: Period length in seconds.
            burst: Maximum number of unused tokens that can accumulate.

        Raises:
            ValueError: If ``rate``, ``per``, or ``burst`` is not positive.

        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        if per <= 0:
            raise ValueError("per must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")

        self._seconds_per_token = per / rate
        self._capacity = float(burst)
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._clock: Callable[[], float] = time.monotonic
        self._sleep: SleepCallable = asyncio.sleep
        self._lock = threading.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        """Wait until ``cost`` operation tokens have been reserved.

        Args:
            cost: Number of operation tokens to charge.

        Raises:
            ValueError: If ``cost`` is not positive.

        """
        if cost <= 0:
            raise ValueError("cost must be positive")

        wait_seconds = self._reserve(float(cost))
        if wait_seconds > 0:
            logger.debug(f"waiting {wait_seconds:.2f}")
            await self._sleep(wait_seconds)

    def _reserve(self, cost: float) -> float:
        """Reserve ``cost`` tokens and return the required wait in seconds."""
        with self._lock:
            now = self._clock()
            if now > self._updated_at:
                elapsed = now - self._updated_at
                self._tokens = min(
                    self._capacity,
                    self._tokens + (elapsed / self._seconds_per_token),
                )
                self._updated_at = now

            if self._tokens >= cost:
                self._tokens -= cost
                return 0.0

            missing = cost - self._tokens
            ready_at = self._updated_at + (missing * self._seconds_per_token)
            wait_seconds = ready_at - now
            self._tokens = 0.0
            self._updated_at = ready_at
            return wait_seconds


class RateLimitedStore:
    """Decorate an async object store with logical operation-rate pacing.

    The wrapper implements the async range-read methods used by
    ``async-geotiff`` and delegates all other attributes to the wrapped store.
    Batched range reads are forwarded as one batched call by default and cost
    one token unless a different ``ranges_cost`` policy is provided.
    """

    def __init__(
        self,
        store: Store,
        limiter: ObjectStoreRateLimiter,
        *,
        range_cost: float = 1.0,
        ranges_cost: float | RangeBatchCost = 1.0,
    ) -> None:
        """Create a rate-limited view of ``store``.

        Args:
            store: Store implementing the async range-read contract consumed by
                ``async-geotiff``.
            limiter: Shared limiter to charge before forwarding range reads.
            range_cost: Token cost for each ``get_range_async`` call.
            ranges_cost: Token cost for each ``get_ranges_async`` call, or a
                callable that computes a cost from the path and range lists.

        """
        self.store = store
        self.limiter = limiter
        self.range_cost = range_cost
        self.ranges_cost = ranges_cost

    async def get_range_async(
        self,
        path: str,
        *,
        start: int,
        end: int | None = None,
        length: int | None = None,
    ) -> object:
        """Rate-limit and forward one async range read."""
        await self.limiter.acquire(self.range_cost)
        return await self.store.get_range_async(
            path,
            start=start,
            end=end,
            length=length,
        )

    async def get_ranges_async(
        self,
        path: str,
        *,
        starts: Sequence[int],
        ends: Sequence[int] | None = None,
        lengths: Sequence[int] | None = None,
    ) -> object:
        """Rate-limit and forward one async batched range read."""
        await self.limiter.acquire(self._ranges_cost(path, starts, ends, lengths))
        return await self.store.get_ranges_async(
            path,
            starts=starts,
            ends=ends,
            lengths=lengths,
        )

    def _ranges_cost(
        self,
        path: str,
        starts: Sequence[int],
        ends: Sequence[int] | None,
        lengths: Sequence[int] | None,
    ) -> float:
        """Return the configured token cost for a batched range read."""
        if callable(self.ranges_cost):
            return self.ranges_cost(path, starts, ends, lengths)
        return self.ranges_cost

    def __getattr__(self, name: str) -> object:
        """Delegate unsupported store behavior to the wrapped store."""
        return getattr(self.store, name)