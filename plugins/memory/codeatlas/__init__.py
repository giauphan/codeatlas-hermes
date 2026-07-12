"""CodeAtlas Second Brain — Native MemoryProvider for Hermes Agent.

Registers CodeAtlasMemoryProvider with Hermes's MemoryManager. When
``memory.provider: codeatlas`` is set in config, this provider is
loaded automatically and provides:

  - Auto-retrieval of dreams, genome DNA, immune genes before every turn
  - Auto-saving of valuable knowledge after every turn
  - Native codeatlas tools (query_dreams, search_genome, scan_immune, save)
  - System prompt injection for Second Brain awareness

No user commands required. Entirely automatic.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.codeatlas_provider import CodeAtlasMemoryProvider

log = logging.getLogger(__name__)


def register(ctx: Any) -> None:
    """Register the CodeAtlas Second Brain as a native MemoryProvider."""
    try:
        provider = CodeAtlasMemoryProvider()
        ctx.register_memory_provider(provider)
        log.info("CodeAtlas Second Brain registered as native MemoryProvider")
    except Exception as exc:
        log.warning("Failed to register CodeAtlas provider: %s", exc)
