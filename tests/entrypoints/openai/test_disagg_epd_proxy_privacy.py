import importlib.util
import io
import json
import asyncio
from pathlib import Path

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
                    "meta": {"sha256": "ab" * 32, "num_encoder_token": 64},
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
