# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from cedfs_ec.route_token import RouteTokenError, RouteTokenSigner
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


def _feature(**overrides):
    value = {
        "version": 1,
        "mm_hash": "image-a",
        "model_scope": "scope-a",
        "num_encoder_tokens": 4,
        "tensor_shape": [4, 8],
        "tensor_dtype": "bfloat16",
        "compatibility_fingerprint": "fingerprint-a",
        "ownership_epoch": 7,
        "device_key": "cuda:0",
        "modality": "image",
        "position_offset": 2,
        "position_length": 4,
        "grid_thw": [1, 2, 8],
    }
    value.update(overrides)
    return value


def _request(features=None):
    return ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "describe"}],
            "cedfs_prompt_token_ids": list(range(12)),
            "cedfs_mm_features": features or [_feature()],
        }
    )


def _build(request, q=0.0):
    server = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen2_5_vl",
                vision_config=SimpleNamespace(spatial_merge_size=2),
                text_config=SimpleNamespace(hidden_size=2048),
            ),
            multimodal_config=SimpleNamespace(video_pruning_rate=q),
        )
    )
    return OpenAIServingChat._build_cedfs_native_inputs(server, request)


@pytest.fixture(autouse=True)
def compatibility_context(monkeypatch):
    monkeypatch.setenv("CEDFS_MODEL_SCOPES", "scope-a,scope-b")
    monkeypatch.setenv("CEDFS_COMPATIBILITY_FINGERPRINT", "fingerprint-a")
    monkeypatch.setenv("CEDFS_OWNERSHIP_EPOCH", "7")
    monkeypatch.setenv("CEDFS_DEVICE_KEY", "cuda:0")


def test_native_request_builds_cache_only_engine_input():
    _conversation, engine_inputs = _build(_request())

    assert len(engine_inputs) == 1
    engine_input = engine_inputs[0]
    assert engine_input["prompt_token_ids"] == list(range(12))
    assert engine_input["external_cache_only"] is True
    assert engine_input["mm_hashes"] == {"image": ["image-a"]}
    grid_item = engine_input["mm_kwargs"]["image"][0]
    assert grid_item["image_grid_thw"].data.tolist() == [1, 2, 8]
    assert grid_item["image_grid_thw"].field.keep_on_cpu is True
    assert engine_input["mm_placeholders"]["image"][0].offset == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_scope", "wrong", "model_scope mismatch"),
        ("compatibility_fingerprint", "wrong", "fingerprint mismatch"),
        ("ownership_epoch", 8, "epoch mismatch"),
        ("device_key", "cuda:1", "device_key mismatch"),
    ],
)
def test_native_request_rejects_incompatible_identity(field, value, message):
    with pytest.raises(ValueError, match=message):
        _build(_request([_feature(**{field: value})]))


def test_native_request_rejects_missing_server_context(monkeypatch):
    monkeypatch.delenv("CEDFS_OWNERSHIP_EPOCH")

    with pytest.raises(ValueError, match="context is not configured"):
        _build(_request())


def test_native_protocol_rejects_shape_and_token_mismatch():
    with pytest.raises(ValidationError, match="tensor_shape"):
        _request([_feature(tensor_shape=[4, 0])])
    with pytest.raises(ValidationError, match="position_length"):
        _request([_feature(position_length=3)])


def test_native_request_supports_multi_image_and_repeated_objects():
    request = _request(
        [
            _feature(mm_hash="shared", position_offset=1),
            _feature(mm_hash="shared", position_offset=7),
        ]
    )
    _conversation, engine_inputs = _build(request)

    assert engine_inputs[0]["mm_hashes"] == {"image": ["shared", "shared"]}


def test_native_request_rejects_overlapping_positions():
    with pytest.raises(ValueError, match="overlap"):
        _build(_request([_feature(position_offset=2), _feature(position_offset=4)]))


def test_native_video_passes_grid_timing_and_exact_evs_shape():
    feature = _feature(
        mm_hash="video-a",
        modality="video",
        num_encoder_tokens=494,
        tensor_shape=[494, 2052],
        position_length=494,
        grid_thw=[4, 26, 38],
        second_per_grid_ts=0.5,
        video_pruning_rate=0.5,
    )
    request = _request([feature])
    request.cedfs_prompt_token_ids = list(range(600))

    _conversation, engine_inputs = _build(request, q=0.5)

    assert engine_inputs[0]["mm_hashes"] == {"video": ["video-a"]}
    item = engine_inputs[0]["mm_kwargs"]["video"][0]
    assert item["video_grid_thw"].data.tolist() == [4, 26, 38]
    assert item["second_per_grid_ts"].data.item() == 0.5


def test_native_video_rejects_wrong_q_or_tensor_layout():
    feature = _feature(
        modality="video",
        num_encoder_tokens=494,
        tensor_shape=[494, 2052],
        position_length=494,
        grid_thw=[4, 26, 38],
        second_per_grid_ts=0.5,
        video_pruning_rate=0.5,
    )
    request = _request([feature])
    request.cedfs_prompt_token_ids = list(range(600))
    with pytest.raises(ValueError, match="pruning rate mismatch"):
        _build(request, q=0.75)

    request.cedfs_mm_features[0].tensor_shape = [494, 2048]
    with pytest.raises(ValueError, match="tensor/placeholder mismatch"):
        _build(request, q=0.5)


def test_native_video_v2_checks_frozen_variant(monkeypatch):
    import torch
    from cedfs_ec.evs_object import encode_video_ec_v2

    monkeypatch.setenv("CEDFS_MODEL_REVISION", "weights-a")
    monkeypatch.setenv("CEDFS_PROCESSOR_REVISION", "processor-a")
    monkeypatch.setenv("CEDFS_EVS_VERSION", "evs-a")
    _, descriptor = encode_video_ec_v2(
        "video-a",
        torch.zeros((4, 2052), dtype=torch.bfloat16),
        {
            "model_revision": "weights-a",
            "processor_revision": "processor-a",
            "evs_version": "evs-a",
            "q": 0.5,
            "grid_thw": [2, 4, 4],
            "second_per_grid_ts": 0.25,
            "spatial_merge_size": 2,
            "producer_id": "edge-a",
            "model_scope": "scope-a",
        },
    )
    feature = _feature(
        version=2,
        mm_hash="video-a",
        modality="video",
        num_encoder_tokens=4,
        tensor_shape=[4, 2052],
        grid_thw=[2, 4, 4],
        second_per_grid_ts=0.5,
        video_pruning_rate=0.5,
        semantic_variant=descriptor["semantic_variant"],
        physical_variant=descriptor["physical_variant"],
    )
    # The descriptor above used a different sampling time and must be rejected even
    # though the row count and frozen revisions agree.
    with pytest.raises(ValueError, match="variant mismatch"):
        _build(_request([feature]), q=0.5)


def test_direct_pd_route_token_is_required_and_bound(monkeypatch):
    secret = "0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("CEDFS_REQUIRE_ROUTE_TOKEN", "1")
    monkeypatch.setenv("CEDFS_ROUTE_TOKEN_SECRET", secret)
    monkeypatch.setenv("CEDFS_PD_ID", "1")
    token, _claims = RouteTokenSigner(secret).issue(
        model="test",
        model_scope="scope-a",
        compatibility_fingerprint="fingerprint-a",
        ownership_epoch=7,
        owner_pd="1",
        media_hashes=["image-a"],
    )

    OpenAIServingChat._validate_cedfs_route_token(
        _request(), {"x-cedfs-route-token": token}
    )
    with pytest.raises(RouteTokenError, match="missing"):
        OpenAIServingChat._validate_cedfs_route_token(_request(), {})


def test_direct_pd_rejects_wrong_owner_and_tampering(monkeypatch):
    secret = "0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("CEDFS_REQUIRE_ROUTE_TOKEN", "1")
    monkeypatch.setenv("CEDFS_ROUTE_TOKEN_SECRET", secret)
    monkeypatch.setenv("CEDFS_PD_ID", "0")
    token, _claims = RouteTokenSigner(secret).issue(
        model="test",
        model_scope="scope-a",
        compatibility_fingerprint="fingerprint-a",
        ownership_epoch=7,
        owner_pd="1",
        media_hashes=["image-a"],
    )

    with pytest.raises(RouteTokenError, match="owner_pd mismatch"):
        OpenAIServingChat._validate_cedfs_route_token(
            _request(), {"x-cedfs-route-token": token}
        )
    with pytest.raises(RouteTokenError, match="signature"):
        OpenAIServingChat._validate_cedfs_route_token(
            _request(), {"x-cedfs-route-token": token[:-1] + "A"}
        )
