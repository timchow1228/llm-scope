"""
llm-scope storage — Phase 3: Async SQLite persistence via aiosqlite.

Uses WAL mode + persistent connection for safe concurrent writes.
All writes happen via fire_and_forget tasks so they never block the proxy.
"""

import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import aiosqlite


# ---------------------------------------------------------------------------
# DB path
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    custom = os.environ.get("DEVSCOPE_DB_PATH")
    if custom:
        return Path(custom)
    return Path.home() / ".local" / "share" / "llm-scope" / "calls.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    provider TEXT,
    model TEXT,
    original_model TEXT,
    rewritten_model TEXT,
    endpoint TEXT,
    tag TEXT,
    connect_ms REAL,
    ttft_ms REAL,
    generation_ms REAL,
    total_ms REAL,
    proxy_overhead_ms REAL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cached_tokens INTEGER DEFAULT 0,
    is_estimated INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    status_code INTEGER,
    error TEXT,
    request_body TEXT,
    has_image INTEGER DEFAULT 0,
    payload_size_bytes INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_calls_provider ON calls(provider);
CREATE INDEX IF NOT EXISTS idx_calls_tag ON calls(tag);
"""

# ---------------------------------------------------------------------------
# Global persistent connection
# ---------------------------------------------------------------------------

_db: Optional[aiosqlite.Connection] = None


async def init_db() -> None:
    """Initialize the SQLite DB: create tables, enable WAL mode."""
    global _db
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _db = await aiosqlite.connect(str(db_path))
    _db.row_factory = aiosqlite.Row

    # WAL mode for safe concurrent reads/writes
    await _db.execute("PRAGMA journal_mode=WAL;")
    await _db.execute("PRAGMA busy_timeout=5000;")
    await _db.executescript(_CREATE_TABLE_SQL)
    await _db.commit()

    print(f"  DB ready: {db_path}")


async def close_db() -> None:
    """Gracefully close the persistent DB connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


# ---------------------------------------------------------------------------
# API key / sensitive data scrubbing
# ---------------------------------------------------------------------------

_REDACT_PATTERNS = [
    re.compile(r'sk-[A-Za-z0-9\-_]{8,}', re.IGNORECASE),
    re.compile(r'key-[A-Za-z0-9\-_]{8,}', re.IGNORECASE),
    re.compile(r'Bearer\s+[A-Za-z0-9\-_\.]{8,}', re.IGNORECASE),
]

_BASE64_IMAGE_PATTERN = re.compile(
    r'"data:image/[^;]+;base64,[A-Za-z0-9+/=]{20,}"',
    re.IGNORECASE,
)


def _scrub_sensitive(text: str) -> str:
    """Replace API keys and Base64 images with safe placeholders."""
    # Scrub API keys
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    # Scrub Base64 images (replace with size hint)
    def _replace_image(m: re.Match) -> str:
        size_kb = len(m.group(0)) * 3 // 4 // 1024
        return f'"[BASE64_IMAGE: ~{size_kb}KB]"'
    text = _BASE64_IMAGE_PATTERN.sub(_replace_image, text)
    return text


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

_BODY_LIMIT = int(os.environ.get("DEVSCOPE_BODY_LIMIT", "4000"))


async def save_call(record: dict) -> None:
    """
    INSERT OR REPLACE a call record into SQLite.
    Scrubs sensitive data before writing.
    """
    global _db
    if _db is None:
        return  # DB not initialized — skip silently

    # Scrub and truncate request_body
    body_raw = record.get("request_body", "") or ""
    body_scrubbed = _scrub_sensitive(str(body_raw))
    if _BODY_LIMIT > 0:
        body_scrubbed = body_scrubbed[:_BODY_LIMIT]

    try:
        await _db.execute(
            """
            INSERT OR REPLACE INTO calls (
                id, timestamp, provider, model,
                original_model, rewritten_model,
                endpoint, tag,
                connect_ms, ttft_ms, generation_ms, total_ms, proxy_overhead_ms,
                prompt_tokens, completion_tokens, cached_tokens,
                is_estimated, cost_usd,
                status_code, error,
                request_body, has_image, payload_size_bytes
            ) VALUES (
                :id, :timestamp, :provider, :model,
                :original_model, :rewritten_model,
                :endpoint, :tag,
                :connect_ms, :ttft_ms, :generation_ms, :total_ms, :proxy_overhead_ms,
                :prompt_tokens, :completion_tokens, :cached_tokens,
                :is_estimated, :cost_usd,
                :status_code, :error,
                :request_body, :has_image, :payload_size_bytes
            )
            """,
            {
                "id": record.get("id") or str(uuid.uuid4()),
                "timestamp": record.get("timestamp", 0.0),
                "provider": record.get("provider", ""),
                "model": record.get("model", ""),
                "original_model": record.get("original_model"),
                "rewritten_model": record.get("rewritten_model"),
                "endpoint": record.get("endpoint", "/v1/chat/completions"),
                "tag": record.get("tag"),
                "connect_ms": record.get("connect_ms", 0.0),
                "ttft_ms": record.get("ttft_ms", 0.0),
                "generation_ms": record.get("generation_ms", 0.0),
                "total_ms": record.get("total_ms", 0.0),
                "proxy_overhead_ms": record.get("proxy_overhead_ms", 0.0),
                "prompt_tokens": record.get("prompt_tokens", 0),
                "completion_tokens": record.get("completion_tokens", 0),
                "cached_tokens": record.get("cached_tokens", 0),
                "is_estimated": 1 if record.get("is_estimated") else 0,
                "cost_usd": record.get("cost_usd", 0.0),
                "status_code": record.get("status_code", 200),
                "error": record.get("error"),
                "request_body": body_scrubbed,
                "has_image": 1 if record.get("has_image") else 0,
                "payload_size_bytes": record.get("payload_size_bytes", 0),
            },
        )
        await _db.commit()
    except Exception as exc:
        print(f"⚠ storage: save_call failed: {exc}")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def get_calls(limit: int = 50) -> list[dict]:
    """Return the most recent N call records as a list of dicts."""
    global _db
    if _db is None:
        return []
    try:
        async with _db.execute(
            "SELECT * FROM calls ORDER BY timestamp DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as exc:
        print(f"⚠ storage: get_calls failed: {exc}")
        return []


async def get_call(call_id: str) -> Optional[dict]:
    """Return a single call record by id, or None if not found."""
    global _db
    if _db is None:
        return None
    try:
        async with _db.execute(
            "SELECT * FROM calls WHERE id = ?", (call_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None
    except Exception as exc:
        print(f"⚠ storage: get_call failed: {exc}")
        return None
