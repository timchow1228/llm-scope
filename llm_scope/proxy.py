"""
llm-scope proxy — Phase 4: Multi-provider streaming relay + SQLite persistence + API endpoints.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from llm_scope.providers import (
    resolve_provider,
    calc_cost,
    get_api_key,
    PROVIDERS,
)
from llm_scope.storage import init_db, close_db, save_call, get_calls, get_call


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRIP_REQUEST_HEADERS = {
    "host", "content-length", "transfer-encoding",
    "connection", "keep-alive", "te", "trailers", "upgrade",
}

STRIP_RESPONSE_HEADERS = {
    "content-encoding", "content-length", "transfer-encoding",
}


# ---------------------------------------------------------------------------
# Background task registry (strong refs to prevent GC)
# ---------------------------------------------------------------------------

_background_tasks: set = set()


def fire_and_forget(coro) -> None:
    """Schedule a coroutine as a background task without blocking the caller."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize global httpx client and SQLite DB; tear down on shutdown."""
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=30.0),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
    )
    await init_db()
    yield
    await close_db()
    await app.state.client.aclose()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Active requests counter (module-level int — single-process, single event loop)
# ---------------------------------------------------------------------------

_active_requests: int = 0


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API endpoints — Phase 4
# ---------------------------------------------------------------------------

@app.get("/api/calls")
async def api_get_calls(limit: int = 50):
    """Return the most recent N call records, plus the active request count."""
    rows = await get_calls(limit)
    enriched = []
    for r in rows:
        d = dict(r)
        cached_tok = d.get("cached_tokens") or 0
        if cached_tok > 0:
            _, _, config = resolve_provider(d.get("model", ""))
            in_cost = config.get("input_per_1m", 1.0)
            cache_cost = config.get("cache_per_1m", in_cost)
            d["savings_usd"] = (in_cost - cache_cost) * cached_tok / 1_000_000.0
        else:
            d["savings_usd"] = 0.0
        enriched.append(d)
    return {"calls": enriched, "active_requests": _active_requests}


@app.get("/api/calls/{call_id}/body")
async def api_get_call_body(call_id: str):
    """Return the full request_body for a call. 410 if not found."""
    row = await get_call(call_id)
    if row is None:
        return JSONResponse(status_code=410, content={"error": "Call not found or body was purged"})
    return {"id": call_id, "request_body": row.get("request_body", "")}


@app.get("/api/calls/{call_id}")
async def api_get_call(call_id: str):
    """Return a single call record by id."""
    row = await get_call(call_id)
    if row is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return row


@app.get("/v1/models")
async def list_models():
    """Return an OpenAI-compatible model list for all known providers."""
    model_ids = []
    for provider_config in PROVIDERS.values():
        model_ids.extend(provider_config["models"].keys())
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "llm-scope"}
            for m in model_ids
        ],
    }


@app.get("/dashboard")
async def dashboard():
    """Serve the single-file dashboard HTML."""
    html_path = Path(__file__).parent / "static" / "dashboard.html"
    if not html_path.exists():
        return JSONResponse(status_code=503, content={"error": "Dashboard not built yet"})
    return FileResponse(str(html_path), media_type="text/html")


# ---------------------------------------------------------------------------
# Main proxy endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
@app.post("/tag/{tag}/v1/chat/completions")
async def proxy_chat_completions(request: Request, tag: str = None):
    """
    Proxy POST /v1/chat/completions → upstream LLM.
    Auto-routes by model name, measures timing, persists every call to SQLite.
    """
    global _active_requests
    _active_requests += 1
    t0 = time.perf_counter()
    import time as _time
    ts0 = _time.time()  # Unix timestamp for the record

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        _active_requests -= 1
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON in request body"}},
        )

    model = body.get("model", "")
    provider_name, base_url, model_config = resolve_provider(model)

    # ── Resolve API key ──
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.split(" ", 1)[1]
    else:
        api_key = get_api_key(provider_name)
        if not api_key:
            env_var = PROVIDERS.get(provider_name, {}).get("env_key", "UNKNOWN")
            print(f"✗ [{provider_name}] {model} | No API key — set {env_var}")
            _active_requests -= 1
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"No API key for '{provider_name}'. Set {env_var}."}},
            )

    is_stream = body.get("stream", False)

    # ── Detect image in request ──
    has_image = False
    body_str = json.dumps(body)
    if "image_url" in body_str or "data:image/" in body_str:
        has_image = True

    # ── Inject stream_options ──
    supports_stream_usage = PROVIDERS.get(provider_name, {}).get("supports_stream_usage", False)
    if is_stream and supports_stream_usage:
        body.setdefault("stream_options", {})["include_usage"] = True

    upstream_url = f"{base_url}/v1/chat/completions"

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in STRIP_REQUEST_HEADERS
    }
    headers["authorization"] = f"Bearer {api_key}"

    # ── Shared record template ──
    record_id = str(uuid.uuid4())
    base_record = {
        "id": record_id,
        "timestamp": ts0,
        "provider": provider_name,
        "model": model,
        "original_model": None,
        "rewritten_model": None,
        "endpoint": "/v1/chat/completions",
        "tag": tag or request.headers.get("x-devscope-tag"),
        "has_image": has_image,
        "payload_size_bytes": len(raw_body),
        "request_body": body_str,
    }

    t1 = time.perf_counter()

    try:
        upstream_resp = await request.app.state.client.send(
            request=request.app.state.client.build_request(
                method="POST",
                url=upstream_url,
                headers=headers,
                content=json.dumps(body).encode("utf-8"),
            ),
            stream=True,
        )
    except httpx.ConnectError as exc:
        print(f"✗ [{provider_name}] {model} | Connect error: {exc}")
        _active_requests -= 1
        fire_and_forget(save_call({
            **base_record,
            "connect_ms": (time.perf_counter() - t1) * 1000,
            "ttft_ms": 0.0, "generation_ms": 0.0,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "proxy_overhead_ms": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
            "is_estimated": False, "cost_usd": 0.0,
            "status_code": 502,
            "error": f"Connect error: {exc}",
        }))
        return JSONResponse(status_code=502, content={"error": {"message": str(exc)}})
    except httpx.TimeoutException as exc:
        print(f"✗ [{provider_name}] {model} | Timeout: {exc}")
        _active_requests -= 1
        fire_and_forget(save_call({
            **base_record,
            "connect_ms": (time.perf_counter() - t1) * 1000,
            "ttft_ms": 0.0, "generation_ms": 0.0,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "proxy_overhead_ms": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
            "is_estimated": False, "cost_usd": 0.0,
            "status_code": 504,
            "error": f"Timeout: {exc}",
        }))
        _active_requests -= 1
        return JSONResponse(status_code=504, content={"error": {"message": str(exc)}})

    t2 = time.perf_counter()
    connect_ms = (t2 - t1) * 1000

    # ── Non-streaming ──
    if not is_stream:
        try:
            resp_bytes = await upstream_resp.aread()
        finally:
            await upstream_resp.aclose()

        t4 = time.perf_counter()
        total_ms = (t4 - t0) * 1000

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        cost = 0.0
        status_code = upstream_resp.status_code
        error_msg = None

        try:
            resp_json = json.loads(resp_bytes)
            usage = resp_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cached_tokens = usage.get("prompt_cache_hit_tokens", 0)
            cost = calc_cost(prompt_tokens, completion_tokens, model_config, cached_tokens)
            if status_code != 200:
                ct = upstream_resp.headers.get("content-type", "")
                if "application/json" in ct:
                    error_msg = resp_json.get("error", {}).get("message") or str(status_code)
                else:
                    error_msg = resp_bytes[:120].decode("utf-8", errors="replace").strip()
        except Exception:
            pass

        cost_str = f"${cost:.6f}" if cost > 0.000001 else "$0.00"
        print(
            f"✓ [{provider_name}] {model} | "
            f"TTFT: 0ms (non-stream) | "
            f"Total: {total_ms:.0f}ms | "
            f"Tokens: {prompt_tokens}+{completion_tokens} | "
            f"Cost: {cost_str}"
        )

        fire_and_forget(save_call({
            **base_record,
            "connect_ms": connect_ms,
            "ttft_ms": 0.0,
            "generation_ms": total_ms - connect_ms,
            "total_ms": total_ms,
            "proxy_overhead_ms": 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "is_estimated": False,
            "cost_usd": cost,
            "status_code": status_code,
            "error": error_msg,
        }))

        resp_headers = {k: v for k, v in upstream_resp.headers.items()
                        if k.lower() not in STRIP_RESPONSE_HEADERS}
        _active_requests -= 1
        return StreamingResponse(
            iter([resp_bytes]),
            status_code=status_code,
            headers=resp_headers,
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    # ── Streaming ──

    async def _stream_generator():
        _raw_buf = bytearray()
        t3 = None
        first_content_found = False
        status_code = upstream_resp.status_code
        error_msg = None

        try:
            async for chunk in upstream_resp.aiter_bytes():
                if not first_content_found:
                    try:
                        text = chunk.decode("utf-8", errors="replace")
                        for line in text.split("\n"):
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                try:
                                    d = json.loads(line[6:])
                                    content = (d.get("choices") or [{}])[0].get("delta", {}).get("content", "")
                                    if content:
                                        t3 = time.perf_counter()
                                        first_content_found = True
                                        break
                                except json.JSONDecodeError:
                                    pass
                    except Exception:
                        pass

                yield chunk
                _raw_buf.extend(chunk)

        except asyncio.CancelledError:
            status_code = 499
            error_msg = f"Client disconnected"
            print(f"⚠ [{provider_name}] {model} | Client disconnected")
        except (httpx.ReadError, httpx.RemoteProtocolError) as exc:
            status_code = 502
            error_msg = f"Upstream dropped: {exc}"
            print(f"⚠ [{provider_name}] {model} | {error_msg}")
        finally:
            t4 = time.perf_counter()
            await upstream_resp.aclose()

            total_ms = (t4 - t0) * 1000
            ttft_ms = (t3 - t2) * 1000 if t3 else 0.0
            generation_ms = (t4 - (t3 or t2)) * 1000

            prompt_tokens = 0
            completion_tokens = 0
            cached_tokens = 0
            is_estimated = False

            try:
                decoded = _raw_buf.decode("utf-8", errors="replace")
                # Find last SSE line with usage
                for line in reversed(decoded.split("\n")):
                    if '"usage"' in line and line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            usage = d.get("usage") or {}
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                                cached_tokens = usage.get("prompt_cache_hit_tokens", 0)
                                break
                        except json.JSONDecodeError:
                            pass
                else:
                    # Estimate from collected text
                    collected_text = ""
                    for line in decoded.split("\n"):
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            try:
                                d = json.loads(line[6:])
                                content = (d.get("choices") or [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    collected_text += content
                            except Exception:
                                pass
                    if collected_text:
                        cjk = sum(1 for c in collected_text if '\u4e00' <= c <= '\u9fff')
                        ascii_c = len(collected_text) - cjk
                        completion_tokens = int(cjk * 1.0 + ascii_c / 4)
                        is_estimated = True
            except Exception:
                pass

            cost = calc_cost(prompt_tokens, completion_tokens, model_config, cached_tokens)
            cost_str = f"${cost:.6f}" if cost > 0.000001 else "$0.00"
            tokens_str = (
                f"~{completion_tokens} (est.)" if is_estimated
                else f"{prompt_tokens}+{completion_tokens}"
            )

            print(
                f"✓ [{provider_name}] {model} | "
                f"TTFT: {ttft_ms:.0f}ms | "
                f"Total: {total_ms:.0f}ms | "
                f"Tokens: {tokens_str} | "
                f"Cost: {cost_str}"
            )

            fire_and_forget(save_call({
                **base_record,
                "connect_ms": connect_ms,
                "ttft_ms": ttft_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "proxy_overhead_ms": max(0.0, total_ms - connect_ms - ttft_ms - generation_ms),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "is_estimated": is_estimated,
                "cost_usd": cost,
                "status_code": status_code,
                "error": error_msg,
            }))
            _active_requests -= 1

    resp_headers = {k: v for k, v in upstream_resp.headers.items()
                    if k.lower() not in STRIP_RESPONSE_HEADERS}
    return StreamingResponse(
        _stream_generator(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type="text/event-stream",
    )
