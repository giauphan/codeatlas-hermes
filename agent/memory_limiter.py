"""
Memory limiter for Hermes Python backend.

Sets an OS-level address-space limit (RLIMIT_AS) at startup so the
process crashes safely when it exceeds the budget, and runs periodic
``gc.collect()`` between turns to prevent object accumulation from
long-lived conversation sessions.

Usage:
    from agent.memory_limiter import init_memory_limiter
    init_memory_limiter(max_mb=1024)
"""

from __future__ import annotations

import gc
import logging
import os
import signal
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Track whether we've been initialised so multiple calls are no-ops.
_initialised = False


def init_memory_limiter(max_mb: int = 1024) -> None:
    """Install the memory limiter *once*.

    Must be called early (before any LLM turns) so the RLIMIT and the
    GC timer protect the process from runaway memory usage.

    Parameters
    ----------
    max_mb : int
        Soft limit on address space in megabytes.  The process will
        be killed by the OS when it exceeds this.  Default 1024 (1GB).
    """
    global _initialised
    if _initialised:
        logger.debug("memory_limiter already initialised — skipping")
        return

    _set_address_space_limit(max_mb)
    _start_gc_timer()
    _register_sigusr1_dump()

    _initialised = True
    logger.info("memory_limiter initialised: max=%d MB, gc=60s, SIGUSR1=memory dump", max_mb)


# ── RLIMIT_AS (OS-level fence) ─────────────────────────────────


def _set_address_space_limit(max_mb: int) -> None:
    """Set RLIMIT_AS so the kernel kills the process if it exceeds *max_mb*."""
    try:
        import resource
        limit = max_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        logger.debug("RLIMIT_AS = %d bytes (%d MB)", limit, max_mb)
    except (ImportError, ValueError, PermissionError):
        logger.warning("RLIMIT_AS not available — OS-level memory fence skipped")


# ── Periodic GC thread ─────────────────────────────────────────


def _start_gc_timer(interval_seconds: int = 60) -> None:
    """Start a daemon thread that calls ``gc.collect()`` every *interval*."""

    def _gc_loop():
        while True:
            time.sleep(interval_seconds)
            before = _rss_mb()
            freed = gc.collect()
            after = _rss_mb()
            if freed > 0:
                logger.debug("gc: freed %d objects (%.0f MB → %.0f MB)", freed, before, after)

    t = threading.Thread(target=_gc_loop, daemon=True, name="memory-limiter-gc")
    t.start()
    logger.debug("GC timer started: interval=%ds", interval_seconds)


# ── SIGUSR1 memory dump ────────────────────────────────────────


def _register_sigusr1_dump() -> None:
    """Register SIGUSR1 to print a memory snapshot without killing Hermes."""
    def _dump(signum, frame):
        usage = _memory_summary()
        logger.warning("SIGUSR1 memory dump: %s", usage)
        print(f"\n[memory-dump] {usage}", flush=True)

    try:
        signal.signal(signal.SIGUSR1, _dump)
    except (AttributeError, ValueError):
        pass  # Windows or restricted environment


# ── Helpers ────────────────────────────────────────────────────


def _rss_mb() -> float:
    """Return current RSS in MB (cross-platform)."""
    try:
        # Linux /proc/self/status is the most reliable
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass

    # Fallback: psutil or /proc/self/statm
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    return 0.0


def _memory_summary() -> str:
    """Return a one-line memory summary string."""
    rss = _rss_mb()
    objs = sum(1 for _ in gc.get_objects() if hasattr(_, "__class__"))
    return f"RSS={rss:.0f} MB, objects={objs:,}"


# ── Public API for periodic manual GC ──────────────────────────


def collect_now() -> int:
    """Force an immediate garbage collection and return freed object count."""
    return gc.collect()
