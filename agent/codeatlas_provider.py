"""Native CodeAtlas Second Brain MemoryProvider — 3-Tier Cache Architecture.

Tiers:
  Tier 1 — In-memory memo cache (TTL 30min, query-keyed)
  Tier 2 — Local JSON store (~/.hermes/second_brain/*.json)
  Tier 3 — CodeAtlas Cloud (Oracle 26ai)

Flow:
  prefetch()   → T1(memo) → T2(local) → T3(cloud) → write-through T2
  sync_turn()  → quality filter → T3(cloud FIRST) → T2(local cache)
  queue_prefetch() → background warm T1

This is a Hermes MemoryProvider registered via config memory.provider: codeatlas.
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
UA = "Hermes-CodeAtlas-Provider/2.0"
TIMEOUT = 4
MAX_CONTEXT_CHARS = 2000
MEMO_TTL = 1800  # 30 min in-memory
CACHE_TTL = 86400  # 24h local cache expiry


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


def _local_store_dir() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "second_brain"


class CodeAtlasMemoryProvider(MemoryProvider):
    """Native CodeAtlas Second Brain as a Hermes MemoryProvider — 3-tier cache.

    Tier 1: In-memory memo (TTL 30min, keyed by query hash)
    Tier 2: Local JSON store (~/.hermes/second_brain/*.json)
    Tier 3: CodeAtlas Cloud (Oracle 26ai)

    Automatically retrieves dreams, genome, immune before LLM turns
    and saves valuable knowledge after each turn.
    """

    def __init__(self) -> None:
        self._api_key: str = ""
        self._enabled: bool = False
        self._session_id: str = ""
        # Tier 1: In-memory memo cache
        self._memo_cache: Dict[str, tuple[str, float]] = {}
        self._memo_ttl: float = MEMO_TTL
        # Stats
        self._stats = {"hit_memo": 0, "miss": 0, "hit_cloud": 0, "hit_cache": 0, "saved": 0}

    @property
    def name(self) -> str:
        return "codeatlas"

    def is_available(self) -> bool:
        key = _resolve_api_key()
        if key:
            self._api_key = key
            return True
        store = _local_store_dir()
        return store.exists()

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id or ""
        self._api_key = _resolve_api_key()
        self._enabled = bool(self._api_key)
        store = _local_store_dir()
        store.mkdir(parents=True, exist_ok=True)
        log.info(
            "[CodeAtlas] v2 Initialized — session=%s | cloud=%s | cache=%s",
            self._session_id[:12] if self._session_id else "?",
            "enabled" if self._enabled else "local-only",
            store,
        )

    def system_prompt_block(self) -> str:
        """Inject Second Brain awareness. Tells model not to use SQLite memories."""
        if not self._enabled:
            return ""
        return (
            "## CodeAtlas Second Brain (Active — Primary Memory)\n"
            "Your CodeAtlas Second Brain is your persistent memory. "
            "Relevant dreams, genome DNA, immune genes, and skills are "
            "automatically prefetched and injected before every response.\n"
            "- You do NOT need to ask 'search memories' or 'load dreams'.\n"
            "- After completing a task, knowledge is auto-saved.\n"
            "- Local conversation cache is ephemeral (working memory only).\n"
            "- Use codeatlas tools for deeper retrieval if needed.\n"
        )

    # ── Tier 1: In-memory memo cache ─────────────────────────────────────
    def _memo_get(self, key: str) -> str | None:
        entry = self._memo_cache.get(key)
        if entry and time.time() < entry[1]:
            self._stats["hit_memo"] += 1
            return entry[0]
        return None

    def _memo_set(self, key: str, value: str) -> None:
        now = time.time()
        self._memo_cache[key] = (value, now + self._memo_ttl)
        # Prune expired
        self._memo_cache = {k: v for k, v in self._memo_cache.items() if v[1] > now}

    # ── Tier 2: Local JSON cache ─────────────────────────────────────────
    def _read_cache(self, name: str) -> list[dict]:
        f = _local_store_dir() / f"{name}.json"
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text())
        except Exception:
            return []

    def _write_cache(self, name: str, data: list[dict]) -> None:
        try:
            f = _local_store_dir() / f"{name}.json"
            f.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            pass

    # ── Tier 3: Cloud queries (source of truth) ──────────────────────────
    def _query_dreams(self, query: str, project: str, limit: int = 5) -> tuple[list[dict], bool]:
        r, s = _api_rq("GET", "/api/dreams/query",
                       params={"query": query, "project": project, "limit": limit},
                       api_key=self._api_key)
        if 200 <= s < 300:
            mems = r.get("memories", [])
            if mems:
                self._write_cache("dreams_cache", mems)
                return mems, True
        return [], False

    def _query_genome(self, query: str, project: str) -> tuple[list[dict], bool]:
        r, s = _api_rq("GET", "/api/genome/search",
                       params={"query": query, "project": project, "limit": 3},
                       api_key=self._api_key)
        if 200 <= s < 300:
            genes = r.get("genes", [])
            if genes:
                self._write_cache("genome_cache", genes)
                return genes, True
        return [], False

    def _query_immune(self, problem: str, project: str) -> tuple[str, bool]:
        r, s = _api_rq("GET", "/api/genome/immune/context",
                       params={"problem": problem, "project": project},
                       api_key=self._api_key)
        if 200 <= s < 300:
            ctx = r.get("context", "")
            if ctx and len(ctx) > 50:
                self._write_cache("immune_cache", [{"context": ctx[:1000]}])
                return ctx, True
        return "", False

    # ── Prefetch (called before every LLM turn) ──────────────────────────
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """3-tier cache: memo → cloud (source of truth) → local fallback."""
        if not query:
            return ""

        cache_key = f"pre_{hash(query) % 1000000}"
        project = "hermes-auto"
        parts: list[str] = []

        # Tier 1: Memo cache hit?
        memo_hit = self._memo_get(cache_key)
        if memo_hit:
            return memo_hit

        self._stats["miss"] += 1
        cloud_hit = False

        # Tier 3: Cloud queries (source of truth)
        if self._enabled:
            # Dreams
            mems, ok = self._query_dreams(query, project)
            if ok:
                cloud_hit = True
                lines = ["## 🧠 CodeAtlas Dreams"]
                for m in mems[:5]:
                    c = m.get("content", "")[:300]
                    lines.append(f"- [{m.get('memory_type', '?')}] {c}")
                parts.append("\n".join(lines))
                self._stats["hit_cloud"] += 1

            # Genome DNA
            genes, ok = self._query_genome(query, project)
            if ok:
                cloud_hit = True
                lines = ["## 🧬 Genome DNA"]
                for g in genes[:3]:
                    lines.append(f"- [{g.get('category', '?')}] {g.get('name', '')} "
                                 f"(confidence: {g.get('confidence', '?')})")
                parts.append("\n".join(lines))

            # Immune
            ctx, ok = self._query_immune(query, project)
            if ok:
                cloud_hit = True
                parts.append(f"## 🛡️ Immune Prevention\n{ctx[:500]}")

        # Tier 2: Local cache (fallback when cloud unavailable)
        if not cloud_hit:
            self._stats["hit_cache"] += 1
            for cache_name, label in [
                ("dreams_cache", "## 🧠 Cached Dreams"),
                ("genome_cache", "## 🧬 Cached Genome"),
                ("immune_cache", "## 🛡️ Cached Immune"),
            ]:
                cached = self._read_cache(cache_name)
                if cached:
                    lines = [label]
                    for entry in cached[:3]:
                        c = (entry.get("content") or
                             entry.get("name") or
                             entry.get("context") or "")[:200]
                        if c:
                            lines.append(f"- {c}")
                    parts.append("\n".join(lines))

        if not parts:
            return ""

        combined = "\n\n".join(parts)
        if len(combined) > MAX_CONTEXT_CHARS:
            combined = combined[:MAX_CONTEXT_CHARS] + "\n[...truncated]"

        # Populate Tier 1
        self._memo_set(cache_key, combined)
        return combined

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background warm Tier 1 for the next turn."""
        if not query:
            return
        try:
            self.prefetch(query, session_id=session_id)
        except Exception:
            pass

    # ── Sync (called after every turn) ───────────────────────────────────
    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Write-through save: cloud FIRST (source of truth), then local cache."""
        if not assistant_content or len(assistant_content) < 200:
            return
        if not self._quality_filter(user_content, assistant_content):
            return

        summary = assistant_content[:250].replace("\n", " ").strip()
        dream = {
            "memory_type": "KNOWLEDGE",
            "content": f"[Auto] Q: {user_content[:100]} | A: {summary[:200]}",
            "importance": 5,
            "project": "hermes-auto",
            "session_id": session_id or self._session_id or f"auto-{int(time.time())}",
        }

        # Cloud save FIRST (source of truth)
        cloud_saved = False
        if self._enabled:
            try:
                r, s = _api_rq("POST", "/api/dreams/save", body=dream,
                               api_key=self._api_key)
                if 200 <= s < 300:
                    cloud_saved = True
                    log.info("[CodeAtlas] ☁️ Cloud save OK — id=%s", r.get("id", "?"))
            except Exception as e:
                log.debug("[CodeAtlas] Cloud save failed: %s", e)

        # Local cache always (offline fallback + dedup)
        self._save_local_dream(dream)
        self._stats["saved"] += 1

    def _save_local_dream(self, dream: dict) -> bool:
        """Save to local JSON cache with dedup and pruning."""
        try:
            store = _local_store_dir()
            store.mkdir(parents=True, exist_ok=True)
            f = store / "dreams.json"
            existing: list[dict] = []
            if f.exists():
                existing = json.loads(f.read_text())
            if self._is_duplicate(dream, existing):
                return False
            dream["_local_ts"] = time.time()
            existing.append(dream)
            # Prune: keep last 200, drop >90 days
            cutoff = time.time() - 90 * 86400
            existing = [d for d in existing if d.get("_local_ts", 0) > cutoff]
            existing = existing[-200:]
            f.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
            return True
        except Exception:
            return False

    # ── Tool schemas ──────────────────────────────────────────────────────
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._enabled:
            return []
        return [
            {
                "name": "query_dream_memories",
                "description": "Search CodeAtlas Second Brain for relevant knowledge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "project": {"type": "string", "description": "Project scope (default: hermes-auto)"},
                        "limit": {"type": "integer", "description": "Max results (1-100, default: 5)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_genome",
                "description": "Search CodeAtlas Genome for relevant DNA patterns.",
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
                "description": "Scan CodeAtlas Immune System for known issues.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem": {"type": "string", "description": "Problem description"},
                        "project": {"type": "string", "description": "Project scope"},
                    },
                    "required": ["problem"],
                },
            },
            {
                "name": "save_dream_memory",
                "description": "Save knowledge to CodeAtlas Second Brain.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_type": {"type": "string", "enum": ["KNOWLEDGE", "PATTERN", "MISTAKE", "PREFERENCE", "FEEDBACK"]},
                        "content": {"type": "string", "description": "Knowledge to persist"},
                        "importance": {"type": "integer", "description": "1-10 (default: 5)", "minimum": 1, "maximum": 10},
                        "project": {"type": "string", "description": "Project scope"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                    },
                    "required": ["memory_type", "content"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if not self._enabled:
            return json.dumps({"err": "CodeAtlas not configured"})
        project = args.get("project", "hermes-auto")
        try:
            if tool_name == "query_dream_memories":
                r, s = _api_rq("GET", "/api/dreams/query",
                               params={"query": args.get("query", ""), "project": project,
                                       "limit": args.get("limit", 5)},
                               api_key=self._api_key)
                return json.dumps(r.get("memories", []) if 200 <= s < 300 else {"err": r.get("err", "")})
            elif tool_name == "search_genome":
                r, s = _api_rq("GET", "/api/genome/search",
                               params={"query": args.get("query", ""), "project": project,
                                       "limit": args.get("limit", 5)},
                               api_key=self._api_key)
                return json.dumps(r.get("genes", []) if 200 <= s < 300 else {"err": r.get("err", "")})
            elif tool_name == "scan_immune":
                r, s = _api_rq("GET", "/api/genome/immune",
                               params={"problem": args.get("problem", ""), "project": project},
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
                return json.dumps({"err": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"err": str(e)})

    def shutdown(self) -> None:
        log.info("[CodeAtlas] Shutdown — stats: memo_hit=%d miss=%d cloud=%d cache=%d saved=%d",
                 self._stats["hit_memo"], self._stats["miss"],
                 self._stats["hit_cloud"], self._stats["hit_cache"],
                 self._stats["saved"])
        self._memo_cache.clear()

    # ── Internals ────────────────────────────────────────────────────────
    @staticmethod
    def _quality_filter(user_msg: str, assistant_resp: str) -> bool:
        resp = assistant_resp.strip()
        if len(resp) < 200:
            return False
        signals = ["pattern", "learn", "discover", "fix", "error", "bug", "solution",
                    "architecture", "design", "decision", "implement", "deploy",
                    "config", "remember", "note", "lesson", "convention",
                    "standard", "approach", "workaround", "root cause", "patch", "refactor"]
        if sum(1 for s in signals if s in resp.lower()) < 2:
            return False
        code_lines = sum(1 for line in resp.split("\n")
                         if line.strip().startswith(("```", "import ", "def ", "class ", "const ", "export ")))
        if code_lines > len(resp.split("\n")) * 0.5:
            return False
        return True

    @staticmethod
    def _is_duplicate(dream: dict, existing: list[dict], threshold: float = 0.7) -> bool:
        content = dream.get("content", "").lower().strip()
        if not content:
            return False
        for d in existing[-50:]:
            existing_content = d.get("content", "").lower().strip()
            words_new = set(content.split())
            words_old = set(existing_content.split())
            if not words_new or not words_old:
                continue
            overlap = len(words_new & words_old) / min(len(words_new), len(words_old))
            if overlap > threshold:
                return True
        return False
