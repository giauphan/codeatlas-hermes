"""Native CodeAtlas Second Brain MemoryProvider.

Integrates CodeAtlas as a first-class MemoryProvider registered with
Hermes's MemoryManager. This is NOT a plugin hook — it's the same
integration layer as the built-in memory system and Honcho/Mem0.

Architecture:
  - prefetch()      → auto-retrieve dreams, genome, immune, skills (no user trigger)
  - sync_turn()     → auto-save knowledge, evolve genome (no user command)
  - system_prompt_block() → inject CodeAtlas awareness into system prompt
  - get_tool_schemas()    → expose query/save tools to the model
  - queue_prefetch()      → background prefetch for next turn's context

All retrieval and learning is automatic. The user never needs to say:
  "Load my memories", "Search Dreams", "Query Genome", "Save this".
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8080/"
UA = "Hermes-CodeAtlas-Provider/1.0"
TIMEOUT = 4  # seconds for localhost requests
MAX_CONTEXT_CHARS = 1500  # cap injected context to avoid prompt bloat


def _resolve_api_key() -> str:
    """Resolve CODEATLAS_API_KEY from env or ~/.hermes/.env."""
    key = os.environ.get("CODEATLAS_API_KEY", "")
    if key:
        return key
    env_file = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("CODEATLAS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _api_rq(method: str, path: str, body: dict | None = None,
            params: dict | None = None, api_key: str = "") -> tuple[dict, int]:
    """Call CodeAtlas REST API with proper auth."""
    if not api_key:
        return {"err": "no api key"}, 401
    url = API_URL.rstrip("/") + path
    if params:
        url += "?" + "&".join(
            f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items() if v
        )
    if "apiKey=" not in url:
        sep = "&" if "?" in url else "?"
        url += f"{sep}apiKey={urllib.parse.quote(api_key)}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body else None, method=method
    )
    req.add_header("x-api-key", api_key)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", UA)
    try:
        r = urllib.request.urlopen(req, timeout=TIMEOUT)
        return json.loads(r.read().decode()), r.status
    except urllib.error.HTTPError as e:
        return {"err": e.read().decode("utf-8", errors="replace")[:200]}, e.code
    except Exception as e:
        return {"err": str(e)[:200]}, 0


# ── Local fallback store ──────────────────────────────────────────────────────
def _local_store_dir() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "second_brain"


class CodeAtlasMemoryProvider(MemoryProvider):
    """Native CodeAtlas Second Brain as a Hermes MemoryProvider.

    Automatically:
      - Retrieves relevant dreams, genome DNA, immune genes, and skills
        before every LLM turn (prefetch).
      - Saves valuable new knowledge after every turn (sync_turn).
      - Exposes codeatlas tools (query_dreams, search_genome, etc.) to the model.
      - Injects system prompt awareness so the model knows CodeAtlas is active.

    No user commands needed. Entirely automatic.
    """

    def __init__(self) -> None:
        self._api_key: str = ""
        self._enabled: bool = False
        self._session_id: str = ""
        # Local memo cache: keyed by query hash → (context, expiry)
        self._memo_cache: Dict[str, tuple[str, float]] = {}
        self._memo_ttl: float = 1800.0  # 30 minutes

    # ── MemoryProvider interface ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "codeatlas"

    def is_available(self) -> bool:
        """Available if API key is configured and local store exists."""
        key = _resolve_api_key()
        if key:
            self._api_key = key
            return True
        # Even without API key, local store works
        store = _local_store_dir()
        return store.exists()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Initialize for a session. Called once at agent startup."""
        self._session_id = session_id or ""
        self._api_key = _resolve_api_key()
        self._enabled = bool(self._api_key)
        store = _local_store_dir()
        store.mkdir(parents=True, exist_ok=True)
        log.info(
            "[CodeAtlas] Initialized — session=%s | API=%s | store=%s",
            self._session_id[:12] if self._session_id else "?",
            "enabled" if self._enabled else "local-only",
            store,
        )

    def system_prompt_block(self) -> str:
        """Inject CodeAtlas Second Brain awareness into the system prompt."""
        if not self._enabled:
            return ""
        return (
            "## CodeAtlas Second Brain (Active)\n"
            "Your CodeAtlas Second Brain is connected and active. Before responding:\n"
            "- Relevant dreams, genome DNA, immune genes, and skills are automatically "
            "prefetched and injected into each user message as context.\n"
            "- You do NOT need to ask the user to 'search memories' or 'load dreams' — "
            "retrieval is automatic.\n"
            "- After completing a task, new knowledge is automatically persisted "
            "(no 'save' command needed).\n"
            "- You have access to codeatlas tools (query_dream_memories, search_genome, "
            "scan_immune, save_dream_memory) if deeper retrieval is needed.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Auto-retrieve Second Brain context before the LLM call.

        Returns formatted context string injected into the user message.
        Called automatically by MemoryManager before every turn.
        """
        if not query:
            return ""

        project = "hermes-auto"
        parts: list[str] = []

        # ── Memo cache check ──
        cache_key = f"pre_{hash(query) % 1000000}"
        if cache_key in self._memo_cache:
            ctx, expiry = self._memo_cache[cache_key]
            if time.time() < expiry and ctx:
                log.debug("[CodeAtlas] Memo cache hit (%d chars)", len(ctx))
                return ctx

        # ── Local dreams ──
        try:
            store = _local_store_dir()
            dreams_file = store / "dreams.json"
            if dreams_file.exists():
                local_dreams = json.loads(dreams_file.read_text())
                q_lower = query.lower()
                scored = []
                for d in local_dreams[-100:]:  # Last 100 entries
                    content = d.get("content", "").lower()
                    score = sum(1 for w in q_lower.split() if w in content)
                    if score > 0:
                        scored.append((score, d))
                scored.sort(key=lambda x: (x[0], x[1].get("_local_ts", 0)), reverse=True)
                if scored:
                    lines = ["## 🧠 Cached Second Brain Knowledge"]
                    for s, d in scored[:5]:
                        c = d.get("content", "")[:200]
                        mt = d.get("memory_type", "?")
                        lines.append(f"- [{mt}] {c}")
                    parts.append("\n".join(lines))
        except Exception as e:
            log.debug("[CodeAtlas] Local search failed: %s", e)

        # ── Cloud: Dreams ──
        if self._enabled:
            try:
                r, s = _api_rq("GET", "/api/dreams/query",
                               params={"query": query, "project": project, "limit": 3},
                               api_key=self._api_key)
                if 200 <= s < 300:
                    mems = r.get("memories", [])
                    if mems:
                        lines = ["## ☁️ CodeAtlas Dreams"]
                        for m in mems:
                            c = m.get("content", "")[:200]
                            mt = m.get("memory_type", "?")
                            lines.append(f"- [{mt}] {c}")
                        parts.append("\n".join(lines))
            except Exception as e:
                log.debug("[CodeAtlas] Dream query failed: %s", e)

        # ── Cloud: Genome ──
        if self._enabled:
            try:
                r, s = _api_rq("GET", "/api/genome/search",
                               params={"query": query, "project": project, "limit": 3},
                               api_key=self._api_key)
                if 200 <= s < 300:
                    genes = r.get("genes", [])
                    if genes:
                        lines = ["## 🧬 CodeAtlas Genome DNA"]
                        for g in genes[:3]:
                            lines.append(
                                f"- [{g.get('category', '')}] {g.get('name', '')} "
                                f"(confidence: {g.get('confidence', '')})"
                            )
                        parts.append("\n".join(lines))
            except Exception as e:
                log.debug("[CodeAtlas] Genome search failed: %s", e)

        # ── Cloud: Immune ──
        if self._enabled:
            try:
                r, s = _api_rq("GET", "/api/genome/immune/context",
                               params={"problem": query, "project": project},
                               api_key=self._api_key)
                if 200 <= s < 300:
                    ctx = r.get("context", "")
                    if ctx and len(ctx) > 50:
                        parts.append(
                            f"## 🛡️ Immune Prevention Context\n{ctx[:500]}"
                        )
            except Exception as e:
                log.debug("[CodeAtlas] Immune scan failed: %s", e)

        if not parts:
            return ""

        combined = "\n\n".join(parts)
        # Truncate to avoid prompt bloat
        if len(combined) > MAX_CONTEXT_CHARS:
            combined = combined[:MAX_CONTEXT_CHARS] + "\n[...truncated]"

        # Memo cache
        self._memo_cache[cache_key] = (combined, time.time() + self._memo_ttl)
        # Prune old entries
        now = time.time()
        self._memo_cache = {
            k: v for k, v in self._memo_cache.items()
            if v[1] > now
        }

        return combined

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background prefetch for the next turn."""
        if not query:
            return
        # Warm the memo cache in background
        try:
            self.prefetch(query, session_id=session_id)
        except Exception:
            pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Auto-save valuable knowledge after each turn.

        Quality checks:
          - Response must be substantial (>200 chars)
          - Must contain knowledge signals (patterns, fixes, lessons)
          - Not code-heavy (>50% code lines)
        """
        if not assistant_content or len(assistant_content) < 200:
            return
        if not self._quality_filter(user_content, assistant_content):
            return

        project = "hermes-auto"
        summary = assistant_content[:250].replace("\n", " ").strip()

        dream = {
            "memory_type": "KNOWLEDGE",
            "content": f"[Auto] Q: {user_content[:100]} | A: {summary[:200]}",
            "importance": 5,
            "project": project,
            "session_id": session_id or self._session_id or f"auto-{int(time.time())}",
        }

        # ── Save locally ──
        try:
            store = _local_store_dir()
            store.mkdir(parents=True, exist_ok=True)
            dreams_file = store / "dreams.json"
            existing: list[dict] = []
            if dreams_file.exists():
                existing = json.loads(dreams_file.read_text())
            # Dedup
            if not self._is_duplicate(dream, existing):
                dream["_local_ts"] = time.time()
                existing.append(dream)
                # Prune: keep last 200, drop >90 days old
                cutoff = time.time() - 90 * 86400
                existing = [d for d in existing if d.get("_local_ts", 0) > cutoff]
                existing = existing[-200:]
                dreams_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
                log.info("[CodeAtlas] 💾 Auto-saved to local store")
        except Exception as e:
            log.warning("[CodeAtlas] Local save failed: %s", e)

        # ── Save to cloud ──
        if self._enabled:
            try:
                r, s = _api_rq("POST", "/api/dreams/save", body=dream,
                               api_key=self._api_key)
                if 200 <= s < 300:
                    log.info("[CodeAtlas] ☁️ Auto-saved to cloud")
            except Exception as e:
                log.debug("[CodeAtlas] Cloud save skipped: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose codeatlas tools to the model.

        These are available as native tools — the model can call them
        mid-reasoning when deeper retrieval is needed.
        """
        if not self._enabled:
            return []
        return [
            {
                "name": "query_dream_memories",
                "description": (
                    "Search the CodeAtlas Second Brain dream memory for "
                    "relevant knowledge, patterns, mistakes, and preferences."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "project": {"type": "string", "description": "Project scope (default: hermes-auto)"},
                        "limit": {"type": "integer", "description": "Max results (1-100, default: 5)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_genome",
                "description": (
                    "Search the CodeAtlas Genome for relevant DNA patterns, "
                    "solutions, and architectural knowledge."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "project": {"type": "string", "description": "Project scope"},
                        "limit": {"type": "integer", "description": "Max results (default: 5)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scan_immune",
                "description": (
                    "Scan CodeAtlas Immune System for known issues and "
                    "preventions matching the current problem."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem": {"type": "string", "description": "Description of the problem"},
                        "project": {"type": "string", "description": "Project scope"},
                    },
                    "required": ["problem"],
                },
            },
            {
                "name": "save_dream_memory",
                "description": (
                    "Persist valuable knowledge to the CodeAtlas Second Brain. "
                    "Use this for important learnings, patterns discovered, "
                    "mistakes to avoid, and architectural decisions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "enum": ["KNOWLEDGE", "PATTERN", "MISTAKE", "PREFERENCE", "FEEDBACK"],
                            "description": "Type of memory",
                        },
                        "content": {"type": "string", "description": "The knowledge to persist"},
                        "importance": {
                            "type": "integer",
                            "description": "Importance 1-10 (default: 5)",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "project": {"type": "string", "description": "Project scope"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                    },
                    "required": ["memory_type", "content"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Handle a codeatlas tool call from the model."""
        if not self._enabled:
            return json.dumps({"err": "CodeAtlas not configured"})

        project = args.get("project", "hermes-auto")
        try:
            if tool_name == "query_dream_memories":
                r, s = _api_rq("GET", "/api/dreams/query",
                               params={"query": args.get("query", ""),
                                       "project": project,
                                       "limit": args.get("limit", 5)},
                               api_key=self._api_key)
                return json.dumps(r.get("memories", []) if 200 <= s < 300 else {"err": r.get("err", "")})

            elif tool_name == "search_genome":
                r, s = _api_rq("GET", "/api/genome/search",
                               params={"query": args.get("query", ""),
                                       "project": project,
                                       "limit": args.get("limit", 5)},
                               api_key=self._api_key)
                return json.dumps(r.get("genes", []) if 200 <= s < 300 else {"err": r.get("err", "")})

            elif tool_name == "scan_immune":
                r, s = _api_rq("GET", "/api/genome/immune",
                               params={"problem": args.get("problem", ""),
                                       "project": project},
                               api_key=self._api_key)
                return json.dumps(r.get("genes", []) if 200 <= s < 300 else {"err": r.get("err", "")})

            elif tool_name == "save_dream_memory":
                r, s = _api_rq("POST", "/api/dreams/save", body={
                    "memory_type": args.get("memory_type", "KNOWLEDGE"),
                    "content": args.get("content", ""),
                    "importance": args.get("importance", 5),
                    "project": project,
                    "session_id": args.get("session_id", self._session_id),
                }, api_key=self._api_key)
                return json.dumps({"saved": bool(200 <= s < 300), "id": r.get("id", "")})

            else:
                return json.dumps({"err": f"Unknown codeatlas tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"err": str(e)})

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._memo_cache.clear()

    # ── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _quality_filter(user_msg: str, assistant_resp: str) -> bool:
        """Only persist responses with genuine new knowledge."""
        resp = assistant_resp.strip()
        if len(resp) < 200:
            return False
        knowledge_signals = [
            "pattern", "learn", "discover", "fix", "error", "bug", "solution",
            "architecture", "design", "decision", "implement", "deploy",
            "config", "remember", "note", "lesson", "convention",
            "standard", "approach", "workaround", "root cause", "patch", "refactor",
        ]
        resp_lower = resp.lower()
        if sum(1 for s in knowledge_signals if s in resp_lower) < 2:
            return False
        code_lines = sum(
            1 for line in resp.split("\n")
            if line.strip().startswith(("```", "import ", "def ", "class ", "const ", "export "))
        )
        if code_lines > len(resp.split("\n")) * 0.5:
            return False
        return True

    @staticmethod
    def _is_duplicate(dream: dict, existing: list[dict], threshold: float = 0.7) -> bool:
        """Check Jaccard word overlap for dedup."""
        content = dream.get("content", "").lower().strip()
        if not content:
            return False
        words_new = set(content.split())
        for d in existing[-50:]:
            words_old = set(d.get("content", "").lower().split())
            if not words_new or not words_old:
                continue
            overlap = len(words_new & words_old) / min(len(words_new), len(words_old))
            if overlap > threshold:
                return True
        return False
