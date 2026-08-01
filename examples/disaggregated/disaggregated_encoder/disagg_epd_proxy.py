#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
disagg_encoder_proxy.py

Proxy that routes OpenAI-compatible “/v1/chat/completions” requests to two
clusters:
  • encode  (multimodal feature extraction)
  • decode  (language-model inference)

For MM input we:
    1. Extract *every* image/audio item.
    2. Fire N concurrent requests to the encoder cluster
       (one request per item, with **all text removed**).
    3. Wait for all of them to succeed.
    4. Forward the *original* request to a decode server.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import io
import json
import logging
import os
import random
import uuid
from collections.abc import AsyncIterator
from urllib.request import urlopen

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("proxy")

app = FastAPI()
encode_session: aiohttp.ClientSession | None = None
prefill_session: aiohttp.ClientSession | None = None
decode_session: aiohttp.ClientSession | None = None

###############################################################################
# Utils
###############################################################################


MM_TYPES = {"image_url", "audio_url", "input_audio"}
_BLANK_AUDIO = (
    "data:audio/wav;base64,"
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="
)


def _media_identifier(item: dict) -> str:
    media = item.get(item.get("type"))
    payload = repr(media).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_mm_identifiers(request_data: dict) -> list[str]:
    identifiers = []
    for item in extract_mm_items(request_data):
        identifier = str(item.get("uuid") or _media_identifier(item))
        item["uuid"] = identifier
        identifiers.append(identifier)
    return identifiers


def _log_timeline(req_id: str, event: str, started: float, deadline: float) -> None:
    now = asyncio.get_running_loop().time()
    logger.info(
        "E-PD request timeline: request_id=%s event=%s elapsed_ms=%d "
        "remaining_ms=%d",
        req_id,
        event,
        int((now - started) * 1000),
        max(0, int((deadline - now) * 1000)),
    )


def _minimum_ec_delay_ms(cache_refs: list[dict]) -> float:
    bandwidth_mbps = float(
        getattr(app.state, "simulated_bandwidth_mbps", 0)
    )
    if bandwidth_mbps <= 0:
        return 0.0
    rtt_ms = float(getattr(app.state, "simulated_rtt_ms", 0))
    hold_ms = float(getattr(app.state, "simulated_hold_ms", 0))
    payload_delays = [
        float(ref["payload_size"]) * 8 * 1000
        / (bandwidth_mbps * 1_000_000)
        for ref in cache_refs
        if ref.get("payload_size") is not None
    ]
    if not payload_delays:
        return 0.0
    # Consumer prefetches different MM objects concurrently. The largest object
    # is the lower bound for request-level simulated transfer time.
    return max(payload_delays) + rtt_ms + hold_ms


def ensure_ec_deadline_admission(
    cache_refs: list[dict], req_id: str, deadline: float
) -> None:
    minimum_ec_delay_ms = _minimum_ec_delay_ms(cache_refs)
    if minimum_ec_delay_ms <= 0:
        return
    loop = asyncio.get_running_loop()
    remaining_ms = max(0.0, (deadline - loop.time()) * 1000)
    reserve_ms = float(getattr(app.state, "deadline_reserve_ms", 0))
    if remaining_ms >= minimum_ec_delay_ms + reserve_ms:
        return
    detail = {
        "type": "deadline_admission_reject",
        "request_id": req_id,
        "budget_ms": int(getattr(app.state, "request_timeout_ms", 60000)),
        "remaining_ms": round(remaining_ms, 3),
        "minimum_ec_delay_ms": round(minimum_ec_delay_ms, 3),
        "reserve_ms": round(reserve_ms, 3),
    }
    logger.warning("E-PD deadline admission rejected: %s", detail)
    raise HTTPException(status_code=504, detail=detail)


def _blank_image_url(original: str) -> str:
    if original.startswith("data:"):
        encoded = original.split(",", 1)[1]
        source = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(source))
    else:
        with urlopen(original, timeout=30) as response:  # noqa: S310 - trusted edge
            image = Image.open(io.BytesIO(response.read()))
    blank = Image.new("RGB", image.size)
    output = io.BytesIO()
    blank.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def sanitize_for_pd(request_data: dict, cache_refs: list[dict]) -> dict:
    """Replace raw media with shape-preserving synthetic inputs and cache refs."""
    sanitized = copy.deepcopy(request_data)
    refs = {ref["mm_hash"]: ref for ref in cache_refs}
    for item in extract_mm_items(sanitized):
        identifier = str(item["uuid"])
        if identifier not in refs:
            raise ValueError(f"missing cache_ref for multimodal item {identifier}")
        if item["type"] == "image_url":
            value = item["image_url"]
            url = value["url"] if isinstance(value, dict) else value
            blank_url = _blank_image_url(str(url))
            item["image_url"] = (
                {**value, "url": blank_url} if isinstance(value, dict) else blank_url
            )
        elif item["type"] == "audio_url":
            item["audio_url"] = _BLANK_AUDIO
        else:
            item["input_audio"] = {"data": _BLANK_AUDIO.split(",", 1)[1], "format": "wav"}
    sanitized["cedfs_cache_refs"] = cache_refs
    return sanitized


def _cache_ref(identifier: str, value: dict) -> dict:
    meta = value["meta"]
    return {
        "version": 1,
        "mm_hash": identifier,
        "model_scope": value["model_scope"],
        "sha256": meta["sha256"],
        "num_encoder_token": meta.get("num_encoder_token"),
        "payload_size": meta.get("payload_size"),
        "ino": (value.get("location") or {}).get("ino"),
    }


async def find_ready_cache_refs(identifiers: list[str]) -> dict[str, dict]:
    client = getattr(app.state, "cedfs_client", None)
    scopes = getattr(app.state, "candidate_scopes", [])
    if not identifiers:
        return {}
    if client is None or not scopes:
        raise RuntimeError("CedFS cache_ref resolver is not configured")
    values = await asyncio.to_thread(
        client.batch_find_first_ready, scopes, identifiers, []
    )
    return {
        identifier: _cache_ref(identifier, value)
        for identifier in identifiers
        if (value := values.get(identifier)) is not None
    }


async def resolve_cache_refs(identifiers: list[str]) -> list[dict]:
    if not identifiers:
        return []
    timeout_ms = app.state.cache_ready_timeout_ms
    deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
    pending = list(identifiers)
    resolved = {}
    while pending:
        ready = await find_ready_cache_refs(pending)
        for identifier, cache_ref in ready.items():
            resolved[identifier] = cache_ref
            if identifier in pending:
                pending.remove(identifier)
        if not pending:
            break
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError(
                f"no READY CedFS candidate within {timeout_ms}ms for {pending}"
            )
        await asyncio.sleep(0.02)

    refs = []
    for identifier in identifiers:
        cache_ref = resolved[identifier]
        logger.info(
            "Resolved CedFS cache_ref: mm_hash=%s model_scope=%s",
            identifier,
            cache_ref["model_scope"],
        )
        refs.append(cache_ref)
    return refs


async def send_encoder_request(
    target_url: str, encoder_req: dict, headers: dict[str, str]
) -> tuple[int, str]:
    """Send one encoder request under the process-wide inflight limit."""

    async def _send() -> tuple[int, str]:
        assert encode_session is not None
        async with encode_session.post(
            f"{target_url}/v1/chat/completions",
            json=encoder_req,
            headers=headers,
        ) as response:
            body = await response.read()
            return response.status, body.decode("utf-8", errors="replace")

    semaphore: asyncio.Semaphore | None = app.state.encoder_semaphore
    if semaphore is None:
        return await _send()
    async with semaphore:
        return await _send()


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items


async def fanout_encoder_primer(
    orig_request: dict,
    e_urls: list[str],
    req_id: str,
    identifiers_to_encode: set[str] | None = None,
) -> list[str]:
    """
    1. Build one request *per MM item* with all text removed.
    2. Send them concurrently to the encode cluster.
    3. Raise if any of them fails.
    """
    logger.info("[%s] Processing multimodal items...", req_id)

    mm_items = [
        item
        for item in extract_mm_items(orig_request)
        if identifiers_to_encode is None
        or str(item["uuid"]) in identifiers_to_encode
    ]
    if not mm_items:
        logger.info("[%s] No multimodal items, skipping encoder", req_id)
        return []  # nothing to do

    logger.info("[%s] got %d multimodal items...", req_id, len(mm_items))

    tasks = []

    # Round-robin over encode servers to distribute load a bit
    url_cycle = (e_urls[i % len(e_urls)] for i in range(len(mm_items)))

    for idx, (item, target_url) in enumerate(zip(mm_items, url_cycle)):
        logger.info(
            "[%s] Routing encoder item #%d to %s",
            req_id,
            idx,
            target_url,
        )
        # Derive a *child* request id:  <parent>:<index>:<random-short>
        child_req_id = f"{req_id}:{idx}:{uuid.uuid4().hex[:6]}"
        headers = {"x-request-id": child_req_id}

        encoder_req = {
            # You *may* need to keep additional fields
            "model": orig_request.get("model"),
            "messages": [
                {"role": "user", "content": [item]},
            ],
            # Only need 1 token so the server actually runs the encoder path
            "max_tokens": 1,
            "stream": False,
        }
        tasks.append(send_encoder_request(target_url, encoder_req, headers))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fail fast if any sub-request failed
    for idx, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(
                "[%s] Encoder request #%d raised exception: %s",
                req_id,
                idx,
                r,
                exc_info=r,
            )
            raise HTTPException(
                status_code=502, detail=f"Encoder request failed: {str(r)}"
            )
        status, detail = r
        if status != 200:
            logger.error(
                "[%s] Encoder request #%d returned status %s: %s",
                req_id,
                idx,
                status,
                detail,
            )
            raise HTTPException(
                status_code=status,
                detail=f"Encoder request failed: {detail}",
            )

    logger.info(
        "[%s] All %d encoder requests completed successfully", req_id, len(mm_items)
    )
    return [str(item["uuid"]) for item in mm_items]


async def prepare_multimodal_request(
    req_data: dict,
    e_urls: list[str],
    req_id: str,
    request_started: float,
    deadline: float,
) -> dict:
    identifiers = assign_mm_identifiers(req_data)
    if not identifiers:
        return req_data

    policy = getattr(app.state, "encoder_dispatch_policy", "always")
    ready_refs: dict[str, dict] = {}
    missing = list(identifiers)
    if policy != "always":
        ready_refs = await find_ready_cache_refs(identifiers)
        missing = [
            identifier for identifier in identifiers if identifier not in ready_refs
        ]
        logger.info(
            "Encoder dispatch decision: request_id=%s policy=%s hit_count=%d "
            "miss_count=%d dispatched_count=%d",
            req_id,
            policy,
            len(identifiers) - len(missing),
            len(missing),
            0 if policy == "ready-only" else len(missing),
        )
        _log_timeline(req_id, "cache_lookup_done", request_started, deadline)
        if missing and policy == "ready-only":
            raise HTTPException(
                status_code=424,
                detail={
                    "type": "cedfs_ready_cache_miss",
                    "request_id": req_id,
                    "missing_mm_hashes": missing,
                },
            )

    if missing:
        await fanout_encoder_primer(req_data, e_urls, req_id, set(missing))
        _log_timeline(req_id, "encoder_done", request_started, deadline)
        produced_refs = await resolve_cache_refs(missing)
        ready_refs.update({ref["mm_hash"]: ref for ref in produced_refs})
    else:
        logger.info(
            "[%s] Encoder skipped: policy=%s ready_count=%d",
            req_id,
            policy,
            len(ready_refs),
        )
        _log_timeline(req_id, "encoder_skipped", request_started, deadline)

    cache_refs = [ready_refs[identifier] for identifier in identifiers]
    _log_timeline(req_id, "cache_ref_ready", request_started, deadline)
    ensure_ec_deadline_admission(cache_refs, req_id, deadline)
    return sanitize_for_pd(req_data, cache_refs)


async def maybe_prefill(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """
    - Do prefill-only task if p_url exist;
    - Return modified request data with kv transfer params (for nixl connector)
    - Else, skip and return the original request data for decode
    """
    if p_url:
        logger.info("[%s] Processing through prefill: %s", req_id, p_url)

        prefill_response = await process_prefill_stage(req_data, p_url, req_id)
        # for nixl connector to facilitate kv transfer...
        prefill_response_json = await prefill_response.json()
        kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        return req_data
    else:
        return req_data


async def process_prefill_stage(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """Process request through Prefill stage and return kv_transfer_params"""
    logger.info("[%s] Sending prefill request to: %s", req_id, p_url)

    prefill_request = req_data.copy()
    prefill_request["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill_request["stream"] = False
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1
    if "stream_options" in prefill_request:
        del prefill_request["stream_options"]

    headers = {"x-request-id": req_id}
    try:
        prefill_response = await prefill_session.post(
            f"{p_url}/v1/chat/completions", json=prefill_request, headers=headers
        )
        prefill_response.raise_for_status()

        if prefill_response.status != 200:
            error_text = await prefill_response.text()
            logger.error(
                "[%s] Prefill request failed with status %d: %s",
                req_id,
                prefill_response.status,
                error_text,
            )
            raise HTTPException(
                status_code=prefill_response.status,
                detail={"error": "Prefill request failed", "message": error_text},
            )
        logger.info("[%s] Prefill request completed successfully", req_id)

        return prefill_response

    except Exception as e:
        logger.error("Prefill processing failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Prefill processing error", "message": str(e)},
        ) from e


###############################################################################
# Middleware for request/response logging
###############################################################################


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and responses"""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Log incoming request
    logger.info(
        ">>> [%s] %s %s from %s",
        req_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        # Process request
        response = await call_next(request)

        # Log response
        logger.info(
            "<<< [%s] %s %s completed with status %d",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
        )

        return response
    except Exception as e:
        # Log errors
        logger.exception(
            "!!! [%s] %s %s failed with error: %s",
            req_id,
            request.method,
            request.url.path,
            str(e),
        )
        raise


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global encode_session, prefill_session, decode_session
    timeout = aiohttp.ClientTimeout(total=100_000)
    encode_session = aiohttp.ClientSession(
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=0, force_close=False),
    )
    if app.state.p_urls:
        # only setup if prefill instance(s) exist
        prefill_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=0, force_close=False),
        )
    # A stale keep-alive connection can be closed by the vLLM server between
    # selection and POST. Do not retry a potentially non-idempotent generation;
    # use a fresh decode connection instead.
    decode_session = aiohttp.ClientSession(
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=0, force_close=True),
    )
    max_encoder_inflight = app.state.max_encoder_inflight
    app.state.encoder_semaphore = (
        asyncio.Semaphore(max_encoder_inflight) if max_encoder_inflight > 0 else None
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global encode_session, prefill_session, decode_session
    if encode_session:
        await encode_session.close()
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


###############################################################################
# Core forwarding
###############################################################################


async def forward_non_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    d_url: str,
    request_started: float,
    deadline: float,
) -> dict:
    try:
        async with asyncio.timeout_at(deadline):
            # Step 1: Resolve ready ECs or dispatch only the required Encoder work.
            req_data = await prepare_multimodal_request(
                req_data, e_urls, req_id, request_started, deadline
            )

            # Step 2: Process through Prefill instance
            req_data = await maybe_prefill(req_data, p_url, req_id)

            # Step 3: Process through Decode instance
            logger.info("[%s] Forwarding to decode: %s", req_id, d_url)
            _log_timeline(req_id, "pd_submit", request_started, deadline)
            headers = {"x-request-id": req_id}

            # Cancelling this context closes the in-flight aiohttp request. The
            # vLLM API server observes the disconnect and aborts request_id.
            async with decode_session.post(
                f"{d_url}/v1/chat/completions", json=req_data, headers=headers
            ) as resp:
                if resp.status >= 400:
                    error_body = await resp.text()
                    raise HTTPException(
                        status_code=resp.status,
                        detail={
                            "type": "pd_request_failed",
                            "request_id": req_id,
                            "message": error_body,
                        },
                    )
                result = await resp.json()
            _log_timeline(req_id, "response_done", request_started, deadline)
            return result

    except HTTPException:
        _log_timeline(req_id, "terminal_error", request_started, deadline)
        raise
    except TimeoutError as exc:
        _log_timeline(req_id, "deadline_exceeded", request_started, deadline)
        raise HTTPException(
            status_code=504,
            detail={
                "type": "request_deadline_exceeded",
                "request_id": req_id,
                "budget_ms": int(app.state.request_timeout_ms),
            },
        ) from exc
    except Exception as e:
        _log_timeline(req_id, "terminal_error", request_started, deadline)
        logger.exception("[%s] Error in forward_non_stream: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


async def forward_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    d_url: str,
    request_started: float,
    deadline: float,
) -> AsyncIterator[str]:
    try:
        async with asyncio.timeout_at(deadline):
            # Step 1: Resolve ready ECs or dispatch only the required Encoder work.
            req_data = await prepare_multimodal_request(
                req_data, e_urls, req_id, request_started, deadline
            )

            # Step 2: Process through Prefill instance
            req_data = await maybe_prefill(req_data, p_url, req_id)

            # Step 3: Process through Decode instance
            logger.info("[%s] Starting streaming from decode: %s", req_id, d_url)
            _log_timeline(req_id, "pd_submit", request_started, deadline)
            headers = {"x-request-id": req_id}

            async with decode_session.post(
                f"{d_url}/v1/chat/completions",
                json=req_data,
                headers=headers,
            ) as resp:
                if resp.status >= 400:
                    error_body = await resp.text()
                    raise HTTPException(
                        status_code=resp.status,
                        detail={
                            "type": "pd_request_failed",
                            "request_id": req_id,
                            "message": error_body,
                        },
                    )
                async for chunk in resp.content.iter_chunked(1024):
                    if chunk:
                        yield chunk.decode("utf-8", errors="ignore")

            logger.info("[%s] Streaming completed", req_id)
            _log_timeline(req_id, "response_done", request_started, deadline)

    except HTTPException as exc:
        _log_timeline(req_id, "terminal_error", request_started, deadline)
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"type": "proxy_http_error", "message": str(exc.detail)}
        )
        yield (
            f"data: {json.dumps({'error': detail})}\n\n"
            "data: [DONE]\n\n"
        )
    except TimeoutError:
        _log_timeline(req_id, "deadline_exceeded", request_started, deadline)
        error = {
            "error": {
                "type": "request_deadline_exceeded",
                "request_id": req_id,
                "budget_ms": int(app.state.request_timeout_ms),
            }
        }
        yield f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n"
    except Exception as e:
        _log_timeline(req_id, "terminal_error", request_started, deadline)
        logger.exception("[%s] Error in forward_stream: %s", req_id, str(e))
        error = {
            "error": {
                "type": "proxy_streaming_error",
                "request_id": req_id,
                "message": str(e),
            }
        }
        yield f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n"


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        request_started = asyncio.get_running_loop().time()
        deadline = request_started + app.state.request_timeout_ms / 1000
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        _log_timeline(req_id, "request_ingress", request_started, deadline)

        # Rotate the list once per request so a single-image workload uses all
        # Encoder instances instead of always selecting the first one.
        encoder_rr_index = getattr(app.state, "encoder_rr_index", 0)
        encoder_index = encoder_rr_index % len(app.state.e_urls)
        app.state.encoder_rr_index = encoder_rr_index + 1
        e_urls = (
            app.state.e_urls[encoder_index:]
            + app.state.e_urls[:encoder_index]
        )
        p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
        decode_rr_index = getattr(app.state, "decode_rr_index", 0)
        decode_index = decode_rr_index % len(app.state.d_urls)
        app.state.decode_rr_index = decode_rr_index + 1
        d_url = app.state.d_urls[decode_index]
        logger.info(
            "[%s] Routing request with encoder_policy=%s encoder_start=%s decode=%s",
            req_id,
            app.state.encoder_dispatch_policy,
            e_urls[0],
            d_url,
        )

        is_streaming = req_data.get("stream", False)

        if is_streaming:
            return StreamingResponse(
                forward_stream(
                    req_data,
                    req_id,
                    e_urls,
                    p_url,
                    d_url,
                    request_started,
                    deadline,
                ),
                media_type="text/event-stream",
            )
        result = await forward_non_stream(
            req_data,
            req_id,
            e_urls,
            p_url,
            d_url,
            request_started,
            deadline,
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat_completions endpoint: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Request processing error: {str(e)}"
        ) from e


@app.get("/v1/models")
async def list_models():
    async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health_check():
    async def healthy(urls):
        if not urls:
            return "empty"
        for u in urls:
            try:
                async with encode_session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            except Exception:
                return "unhealthy"
        return "healthy"

    e_status, p_status, d_status = await asyncio.gather(
        healthy(app.state.e_urls), healthy(app.state.p_urls), healthy(app.state.d_urls)
    )

    overall_healthy = all(
        status != "unhealthy" for status in (e_status, p_status, d_status)
    )

    status_code = 200 if overall_healthy else 503

    return JSONResponse(
        {
            "proxy": "healthy",
            "encode_cluster": e_status,
            "prefill_cluster": p_status,
            "decode_cluster": d_status,
        },
        status_code=status_code,
    )


###############################################################################
# Simple profiler fan-out (unchanged except for sessions)
###############################################################################


async def _post_if_available(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
) -> dict | None:
    """
    POST `payload` to `url`.

    Returns
    -------
    • The decoded JSON body on success (2xx)
    • None if the endpoint does not exist (404)
    • Raises for anything else.
    """
    try:
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:  # profiling disabled on that server
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        # Pass 404 through the branch above, re-raise everything else
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise
    except Exception:
        # Network errors etc.: propagate
        raise


async def _profile_cmd(cmd: str, payload: dict, e_url: str, p_url: str, d_url: str):
    """
    Fire & forget to both clusters, tolerate 404.
    """
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(
        encode_session, f"{e_url}/{cmd}_profile", payload, headers
    )
    prefill_task = (
        _post_if_available(prefill_session, f"{p_url}/{cmd}_profile", payload, headers)
        if p_url is not None
        else asyncio.sleep(0)
    )
    decode_task = _post_if_available(
        decode_session, f"{d_url}/{cmd}_profile", payload, headers
    )

    encode_res, prefill_res, decode_res = await asyncio.gather(
        encode_task, prefill_task, decode_task
    )

    # If *all* clusters said “I don’t have that route”, surface an error
    if encode_res is prefill_res is decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on all clusters",
        )

    return {
        "encode": encode_res,  # may be None
        "prefill": prefill_res,  # may be None
        "decode": decode_res,  # may be None
    }


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("start", body, e_url, p_url, d_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("stop", body, e_url, p_url, d_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-servers-urls",
        required=True,
        help=(
            'Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") ',
            'to enable E->P->D, set "disable" or "none" to enable E->PD',
        ),
    )
    parser.add_argument(
        "--decode-servers-urls",
        required=True,
        help='Comma-separated decode URLs ("http://d1:8005,http://d2:8006")',
    )
    parser.add_argument(
        "--max-encoder-inflight",
        type=int,
        default=0,
        help="Maximum concurrent Proxy-to-Encoder requests (0 means unlimited)",
    )
    parser.add_argument(
        "--encoder-dispatch-policy",
        choices=("always", "lookup-first", "ready-only"),
        default="always",
        help=(
            "Encoder dispatch policy: always preserves the historical path; "
            "lookup-first encodes only CedFS misses; ready-only fails on any miss"
        ),
    )
    parser.add_argument(
        "--cedfs-config",
        help="CedFS client TOML used by the proxy-side cache_ref resolver",
    )
    parser.add_argument(
        "--candidate-scopes",
        default="",
        help="Ordered comma-separated compatible producer scopes",
    )
    parser.add_argument(
        "--cache-ready-timeout-ms",
        type=int,
        default=60000,
        help="Bounded wait for COMPLETE CedFS objects before terminal failure",
    )
    parser.add_argument(
        "--request-timeout-ms",
        type=int,
        default=60000,
        help="End-to-end Proxy request deadline",
    )
    parser.add_argument("--simulated-bandwidth-mbps", type=float, default=0)
    parser.add_argument("--simulated-rtt-ms", type=float, default=0)
    parser.add_argument("--simulated-hold-ms", type=float, default=0)
    parser.add_argument(
        "--deadline-reserve-ms",
        type=int,
        default=10000,
        help="Budget reserved for PD processing and cancellation cleanup",
    )

    args = parser.parse_args()
    app.state.e_urls = [
        u.strip() for u in args.encode_servers_urls.split(",") if u.strip()
    ]
    app.state.d_urls = [
        u.strip() for u in args.decode_servers_urls.split(",") if u.strip()
    ]
    app.state.max_encoder_inflight = max(0, args.max_encoder_inflight)
    app.state.encoder_dispatch_policy = args.encoder_dispatch_policy
    app.state.encoder_rr_index = 0
    app.state.decode_rr_index = 0
    app.state.candidate_scopes = [
        scope.strip() for scope in args.candidate_scopes.split(",") if scope.strip()
    ]
    app.state.cache_ready_timeout_ms = max(1, args.cache_ready_timeout_ms)
    app.state.request_timeout_ms = max(1, args.request_timeout_ms)
    app.state.simulated_bandwidth_mbps = max(
        0.0, args.simulated_bandwidth_mbps
    )
    app.state.simulated_rtt_ms = max(0.0, args.simulated_rtt_ms)
    app.state.simulated_hold_ms = max(0.0, args.simulated_hold_ms)
    app.state.deadline_reserve_ms = max(0, args.deadline_reserve_ms)
    app.state.cedfs_client = None
    if args.cedfs_config:
        from cedfs_ec._native import CedfsECClient

        app.state.cedfs_client = CedfsECClient(args.cedfs_config)
    # handle prefill instances
    if args.prefill_servers_urls.lower() in ("disable", "none", ""):
        app.state.p_urls = []
        logger.info(
            "Disaggregated prefill phase explicitly disabled by user. Running E + PD..."
        )
    else:
        app.state.p_urls = [
            u.strip() for u in args.prefill_servers_urls.split(",") if u.strip()
        ]
        logger.info("Disaggregated prefill phase is enabled. Running E + P + D...")

    logger.info("Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers: %s", app.state.e_urls)
    logger.info("Encoder dispatch policy: %s", app.state.encoder_dispatch_policy)
    logger.info("Maximum encoder inflight: %s", app.state.max_encoder_inflight or "unlimited")
    logger.info("Prefill instances %s", app.state.p_urls)
    logger.info("Decode servers: %s", app.state.d_urls)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=True,
    )
