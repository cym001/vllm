import importlib.util
import io
import json
import asyncio
from pathlib import Path

import pytest
from PIL import Image


PROXY = (
    Path(__file__).parents[3]
    / "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"
)
SPEC = importlib.util.spec_from_file_location("disagg_epd_proxy", PROXY)
proxy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(proxy)


def _image_data_url() -> str:
    import base64

    output = io.BytesIO()
    Image.new("RGB", (19, 13), "red").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_pd_request_has_cache_ref_but_not_original_media():
    original_url = _image_data_url()
    request = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": original_url}},
                ],
            }
        ],
    }
    identifiers = proxy.assign_mm_identifiers(request)
    ref = {
        "version": 1,
        "mm_hash": identifiers[0],
        "model_scope": "rg/fp/edge-a",
        "sha256": "ab" * 32,
        "num_encoder_token": 64,
    }
    sanitized = proxy.sanitize_for_pd(request, [ref])
    serialized = json.dumps(sanitized)

    assert original_url not in serialized
    assert sanitized["cedfs_cache_refs"] == [ref]
    item = proxy.extract_mm_items(sanitized)[0]
    assert item["uuid"] == identifiers[0]
    blank_url = item["image_url"]["url"]
    assert blank_url.startswith("data:image/png;base64,")


def test_cache_ref_resolution_waits_until_ready():
    class DelayedClient:
        def __init__(self):
            self.calls = 0

        def batch_find_first_ready(self, scopes, identifiers, excluded):
            self.calls += 1
            if self.calls == 1:
                return {identifiers[0]: None}
            return {
                identifiers[0]: {
                    "model_scope": scopes[0],
                    "meta": {
                        "sha256": "ab" * 32,
                        "num_encoder_token": 64,
                        "payload_size": 257_359_872,
                    },
                    "location": {"ino": 9},
                }
            }

    client = DelayedClient()
    proxy.app.state.cedfs_client = client
    proxy.app.state.candidate_scopes = ["rg_fp_edge"]
    proxy.app.state.cache_ready_timeout_ms = 1000

    refs = asyncio.run(proxy.resolve_cache_refs(["image-1"]))

    assert client.calls == 2
    assert refs[0]["model_scope"] == "rg_fp_edge"
    assert refs[0]["ino"] == 9
    assert refs[0]["payload_size"] == 257_359_872


def _cache_ref(mm_hash: str) -> dict:
    return {
        "version": 1,
        "mm_hash": mm_hash,
        "model_scope": "rg_fp_edge",
        "sha256": "ab" * 32,
        "num_encoder_token": 64,
        "payload_size": 1024,
        "ino": 9,
    }


def test_ready_only_hit_skips_encoder(monkeypatch):
    request = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url()}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
    }
    identifier = proxy.assign_mm_identifiers(request)[0]
    encoder_called = False

    async def find_ready(identifiers):
        return {identifier: _cache_ref(identifier)}

    async def dispatch(*_args, **_kwargs):
        nonlocal encoder_called
        encoder_called = True
        return []

    monkeypatch.setattr(proxy, "find_ready_cache_refs", find_ready)
    monkeypatch.setattr(proxy, "fanout_encoder_primer", dispatch)
    monkeypatch.setattr(
        proxy.app.state, "encoder_dispatch_policy", "ready-only", raising=False
    )
    proxy.app.state.simulated_bandwidth_mbps = 0

    async def check():
        loop = asyncio.get_running_loop()
        started = loop.time()
        return await proxy.prepare_multimodal_request(
            request, ["http://encoder"], "ready-hit", started, started + 60
        )

    prepared = asyncio.run(check())

    assert encoder_called is False
    assert prepared["cedfs_cache_refs"] == [_cache_ref(identifier)]


def test_ready_only_miss_fails_without_encoder(monkeypatch):
    request = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url()}},
                ],
            }
        ],
    }
    encoder_called = False

    async def find_ready(_identifiers):
        return {}

    async def dispatch(*_args, **_kwargs):
        nonlocal encoder_called
        encoder_called = True
        return []

    monkeypatch.setattr(proxy, "find_ready_cache_refs", find_ready)
    monkeypatch.setattr(proxy, "fanout_encoder_primer", dispatch)
    monkeypatch.setattr(
        proxy.app.state, "encoder_dispatch_policy", "ready-only", raising=False
    )

    async def check():
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(proxy.HTTPException) as exc_info:
            await proxy.prepare_multimodal_request(
                request, ["http://encoder"], "ready-miss", started, started + 60
            )
        assert exc_info.value.status_code == 424
        assert exc_info.value.detail["type"] == "cedfs_ready_cache_miss"

    asyncio.run(check())
    assert encoder_called is False


def test_lookup_first_dispatches_only_missing_items(monkeypatch):
    request = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url()},
                        "uuid": "ready-image",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url()},
                        "uuid": "missing-image",
                    },
                ],
            }
        ],
    }
    dispatched = None

    async def find_ready(_identifiers):
        return {"ready-image": _cache_ref("ready-image")}

    async def dispatch(_request, _urls, _req_id, identifiers_to_encode):
        nonlocal dispatched
        dispatched = identifiers_to_encode
        return list(identifiers_to_encode)

    async def resolve(identifiers):
        return [_cache_ref(identifier) for identifier in identifiers]

    monkeypatch.setattr(proxy, "find_ready_cache_refs", find_ready)
    monkeypatch.setattr(proxy, "fanout_encoder_primer", dispatch)
    monkeypatch.setattr(proxy, "resolve_cache_refs", resolve)
    monkeypatch.setattr(
        proxy.app.state, "encoder_dispatch_policy", "lookup-first", raising=False
    )
    proxy.app.state.simulated_bandwidth_mbps = 0

    async def check():
        loop = asyncio.get_running_loop()
        started = loop.time()
        return await proxy.prepare_multimodal_request(
            request, ["http://encoder"], "partial-hit", started, started + 60
        )

    prepared = asyncio.run(check())

    assert dispatched == {"missing-image"}
    assert [ref["mm_hash"] for ref in prepared["cedfs_cache_refs"]] == [
        "ready-image",
        "missing-image",
    ]


def test_always_policy_preserves_encoder_first_path(monkeypatch):
    request = {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url()},
                        "uuid": "image-a",
                    },
                ],
            }
        ],
    }
    lookup_called = False
    dispatched = None

    async def find_ready(_identifiers):
        nonlocal lookup_called
        lookup_called = True
        return {}

    async def dispatch(_request, _urls, _req_id, identifiers_to_encode):
        nonlocal dispatched
        dispatched = identifiers_to_encode
        return ["image-a"]

    async def resolve(identifiers):
        return [_cache_ref(identifier) for identifier in identifiers]

    monkeypatch.setattr(proxy, "find_ready_cache_refs", find_ready)
    monkeypatch.setattr(proxy, "fanout_encoder_primer", dispatch)
    monkeypatch.setattr(proxy, "resolve_cache_refs", resolve)
    monkeypatch.setattr(
        proxy.app.state, "encoder_dispatch_policy", "always", raising=False
    )
    proxy.app.state.simulated_bandwidth_mbps = 0

    async def check():
        loop = asyncio.get_running_loop()
        started = loop.time()
        return await proxy.prepare_multimodal_request(
            request, ["http://encoder"], "always", started, started + 60
        )

    prepared = asyncio.run(check())

    assert lookup_called is False
    assert dispatched == {"image-a"}
    assert prepared["cedfs_cache_refs"] == [_cache_ref("image-a")]


def test_formula_delay_admission_rejects_request_that_cannot_meet_deadline():
    proxy.app.state.request_timeout_ms = 60_000
    proxy.app.state.simulated_bandwidth_mbps = 200
    proxy.app.state.simulated_rtt_ms = 40
    proxy.app.state.simulated_hold_ms = 30_000
    proxy.app.state.deadline_reserve_ms = 10_000
    cache_refs = [{"payload_size": 257_359_872}]

    async def check():
        loop = asyncio.get_running_loop()
        with pytest.raises(proxy.HTTPException) as exc_info:
            proxy.ensure_ec_deadline_admission(
                cache_refs, "large-request", loop.time() + 45
            )
        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["type"] == "deadline_admission_reject"
        assert exc_info.value.detail["minimum_ec_delay_ms"] == pytest.approx(
            40_334.395, abs=0.001
        )

    asyncio.run(check())


def test_formula_delay_admission_accepts_request_with_sufficient_budget():
    proxy.app.state.request_timeout_ms = 60_000
    proxy.app.state.simulated_bandwidth_mbps = 200
    proxy.app.state.simulated_rtt_ms = 40
    proxy.app.state.simulated_hold_ms = 30_000
    proxy.app.state.deadline_reserve_ms = 10_000
    cache_refs = [{"payload_size": 257_359_872}]

    async def check():
        loop = asyncio.get_running_loop()
        proxy.ensure_ec_deadline_admission(
            cache_refs, "large-request", loop.time() + 51
        )

    asyncio.run(check())


def test_non_stream_deadline_cancels_inflight_decode_request():
    class Response:
        status = 200

        def __init__(self):
            self.cancelled = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    response = Response()

    class Session:
        def post(self, *_args, **_kwargs):
            return response

    proxy.decode_session = Session()
    proxy.app.state.request_timeout_ms = 10

    async def check():
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(proxy.HTTPException) as exc_info:
            await proxy.forward_non_stream(
                {},
                "deadline-request",
                ["http://encoder"],
                None,
                "http://decode",
                started,
                started + 0.01,
            )
        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["type"] == "request_deadline_exceeded"

    asyncio.run(check())
    assert response.cancelled is True


def test_non_stream_preserves_pd_terminal_error_body():
    class Response:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def text(self):
            return '{"error":{"type":"engine_error"}}'

    class Session:
        def post(self, *_args, **_kwargs):
            return Response()

    proxy.decode_session = Session()
    proxy.app.state.request_timeout_ms = 60_000

    async def check():
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(proxy.HTTPException) as exc_info:
            await proxy.forward_non_stream(
                {},
                "pd-error-request",
                ["http://encoder"],
                None,
                "http://decode",
                started,
                started + 60,
            )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == {
            "type": "pd_request_failed",
            "request_id": "pd-error-request",
            "message": '{"error":{"type":"engine_error"}}',
        }

    asyncio.run(check())
