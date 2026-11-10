"""Thread-safe circuit breaker shared by LLM providers in one experiment."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Deque, Optional, Tuple

from .base import LLMError


logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class LLMCircuitBreakerFatalError(LLMError):
    """The shared LLM service did not recover before the maximum pause."""

    def __init__(self, paused_seconds: float, max_pause: float) -> None:
        self.paused_seconds = float(paused_seconds)
        self.max_pause = float(max_pause)
        self.resume_checkpoint_path: Optional[str] = None
        self.resume_epoch = 0
        super().__init__(
            "LLM circuit breaker exceeded maximum pause "
            f"({self.paused_seconds:.1f}s >= {self.max_pause:.1f}s)"
        )


def _error_chain(error: BaseException) -> str:
    parts: list[str] = []
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts).lower()


def get_http_status(error: BaseException) -> Optional[int]:
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(current, "status_code", None)
        try:
            if status is not None:
                return int(status)
        except (TypeError, ValueError):
            pass
        current = current.__cause__ or current.__context__
    return None


def classify_breaker_failure(error: BaseException) -> Optional[str]:
    """Return a breaker-counted failure kind, or None for ordinary errors."""

    status = get_http_status(error)
    if status == 429:
        return None
    text = _error_chain(error)
    if status == 503:
        return "http_503"
    if "model_not_found" in text or "model not found" in text:
        return "model_not_found"
    if "no available channel" in text:
        return "no_available_channel"
    if "inference abort" in text:
        return "inference_abort"
    if "buffer overflow" in text:
        return "buffer_overflow"
    if (
        "timeout" in text
        or "timed out" in text
        or any(
            name in text
            for name in ("apitimeouterror", "readtimeout", "connecttimeout")
        )
    ):
        return "timeout"
    return None


class LLMCircuitBreaker:
    """Coordinate service pauses and health probes across provider threads."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        failure_threshold: int = 3,
        failure_window: float = 30.0,
        probe_interval: float = 300.0,
        confirm_interval: float = 30.0,
        probe_successes: int = 2,
        max_pause: float = 3600.0,
        retry_delays: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self.failure_threshold = max(1, int(failure_threshold))
        self.failure_window = max(0.0, float(failure_window))
        self.probe_interval = max(0.0, float(probe_interval))
        self.confirm_interval = max(0.0, float(confirm_interval))
        self.probe_successes = max(1, int(probe_successes))
        self.max_pause = max(0.0, float(max_pause))
        delays = tuple(max(0.0, float(value)) for value in retry_delays)
        self.retry_delays = delays or (1.0, 2.0, 5.0, 10.0)
        self._clock = clock
        self._condition = threading.Condition()
        self._state = CircuitBreakerState.CLOSED
        self._failures: Deque[Tuple[float, str]] = deque()
        self._opened_at: Optional[float] = None
        self._next_probe_at = 0.0
        self._probe_inflight = False
        self._consecutive_probe_successes = 0
        self._open_count = 0
        self._probe_count = 0
        self._probe_success_count = 0
        self._probe_failure_count = 0

    @property
    def state(self) -> CircuitBreakerState:
        with self._condition:
            return self._state

    @property
    def is_blocking(self) -> bool:
        return self.enabled and self.state is not CircuitBreakerState.CLOSED

    def _prune_failures(self, now: float) -> None:
        cutoff = now - self.failure_window
        while self._failures and self._failures[0][0] < cutoff:
            self._failures.popleft()

    def record_success(self) -> None:
        """Record a real-request success without erasing concurrent failures."""

        if not self.enabled:
            return
        with self._condition:
            self._prune_failures(self._clock())

    def record_failure(self, error: BaseException) -> Optional[str]:
        """Count a qualifying service failure and open at the configured threshold."""

        if not self.enabled:
            return None
        kind = classify_breaker_failure(error)
        if kind is None:
            return None
        with self._condition:
            now = self._clock()
            self._prune_failures(now)
            self._failures.append((now, kind))
            if (
                self._state is CircuitBreakerState.CLOSED
                and len(self._failures) >= self.failure_threshold
            ):
                self._state = CircuitBreakerState.OPEN
                self._opened_at = now
                self._next_probe_at = now + self.probe_interval
                self._consecutive_probe_successes = 0
                self._open_count += 1
                logger.error(
                    "LLM circuit breaker OPEN failures=%d window=%.1fs kinds=%s",
                    len(self._failures),
                    self.failure_window,
                    [failure_kind for _, failure_kind in self._failures],
                )
                self._condition.notify_all()
        return kind

    def _raise_if_expired(self, now: float) -> None:
        if self._opened_at is None:
            return
        paused = now - self._opened_at
        if paused >= self.max_pause:
            self._condition.notify_all()
            raise LLMCircuitBreakerFatalError(paused, self.max_pause)

    def wait_until_available(self, health_probe: Callable[[], bool]) -> bool:
        """Block a real request while open; one waiter performs each probe.

        Returns True when this call observed a pause, allowing callers to reset
        their short-retry budget after service recovery.
        """

        if not self.enabled:
            return False
        waited = False
        while True:
            should_probe = False
            with self._condition:
                if self._state is CircuitBreakerState.CLOSED:
                    return waited
                waited = True
                now = self._clock()
                self._raise_if_expired(now)
                if not self._probe_inflight and now >= self._next_probe_at:
                    self._probe_inflight = True
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._probe_count += 1
                    should_probe = True
                else:
                    pause_deadline = (
                        self._opened_at
                        if self._opened_at is not None
                        else now
                    ) + self.max_pause
                    wait_until = (
                        pause_deadline
                        if self._probe_inflight
                        else min(self._next_probe_at, pause_deadline)
                    )
                    timeout = max(0.001, wait_until - now)
                    self._condition.wait(timeout=timeout)
            if not should_probe:
                continue

            probe_ok = False
            try:
                probe_ok = bool(health_probe())
            except Exception as exc:
                logger.warning("LLM circuit breaker health probe failed: %s", exc)

            with self._condition:
                now = self._clock()
                self._probe_inflight = False
                if probe_ok:
                    self._probe_success_count += 1
                    self._consecutive_probe_successes += 1
                    if self._consecutive_probe_successes >= self.probe_successes:
                        paused = now - (
                            self._opened_at
                            if self._opened_at is not None
                            else now
                        )
                        self._state = CircuitBreakerState.CLOSED
                        self._opened_at = None
                        self._next_probe_at = 0.0
                        self._failures.clear()
                        logger.info(
                            "LLM circuit breaker CLOSED after %.1fs and %d consecutive probes",
                            paused,
                            self._consecutive_probe_successes,
                        )
                        self._consecutive_probe_successes = 0
                    else:
                        self._state = CircuitBreakerState.OPEN
                        self._next_probe_at = now + self.confirm_interval
                        logger.info(
                            "LLM circuit breaker probe succeeded (%d/%d); confirmation in %.1fs",
                            self._consecutive_probe_successes,
                            self.probe_successes,
                            self.confirm_interval,
                        )
                else:
                    self._probe_failure_count += 1
                    self._consecutive_probe_successes = 0
                    self._state = CircuitBreakerState.OPEN
                    self._next_probe_at = now + self.probe_interval
                self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            now = self._clock()
            self._prune_failures(now)
            return {
                "enabled": self.enabled,
                "state": self._state.value,
                "recent_failures": len(self._failures),
                "open_count": self._open_count,
                "probe_count": self._probe_count,
                "probe_success_count": self._probe_success_count,
                "probe_failure_count": self._probe_failure_count,
                "paused_seconds": (
                    max(0.0, now - self._opened_at)
                    if self._opened_at is not None
                    else 0.0
                ),
            }


def build_shared_llm_circuit_breaker(
    config: Any = None,
    *,
    logger_obj: Optional[logging.Logger] = None,
) -> LLMCircuitBreaker:
    """Build one configured breaker for injection into a launcher's providers."""

    defaults = {
        "enabled": True,
        "failure_threshold": 3,
        "failure_window": 30.0,
        "probe_interval": 300.0,
        "confirm_interval": 30.0,
        "probe_successes": 2,
        "max_pause": 3600.0,
        "retry_delays": (1.0, 2.0, 5.0, 10.0),
    }

    def read(name: str) -> Any:
        if config is None:
            return defaults[name]
        if isinstance(config, dict):
            return config.get(name, defaults[name])
        return getattr(config, name, defaults[name])

    breaker = LLMCircuitBreaker(
        enabled=read("enabled"),
        failure_threshold=read("failure_threshold"),
        failure_window=read("failure_window"),
        probe_interval=read("probe_interval"),
        confirm_interval=read("confirm_interval"),
        probe_successes=read("probe_successes"),
        max_pause=read("max_pause"),
        retry_delays=tuple(read("retry_delays")),
    )
    (logger_obj or logger).info(
        "Shared LLM circuit breaker: enabled=%s threshold=%d window=%.1fs "
        "probe=%.1fs confirm=%.1fs successes=%d max_pause=%.1fs",
        breaker.enabled,
        breaker.failure_threshold,
        breaker.failure_window,
        breaker.probe_interval,
        breaker.confirm_interval,
        breaker.probe_successes,
        breaker.max_pause,
    )
    return breaker
