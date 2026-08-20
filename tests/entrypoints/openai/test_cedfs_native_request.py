import pytest
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


@pytest.fixture(autouse=True)
def compatibility_context(monkeypatch):
    monkeypatch.setenv("CEDFS_MODEL_SCOPES", "scope-a,scope-b")
    monkeypatch.setenv("CEDFS_COMPATIBILITY_FINGERPRINT", "fingerprint-a")
    monkeypatch.setenv("CEDFS_OWNERSHIP_EPOCH", "7")
    monkeypatch.setenv("CEDFS_DEVICE_KEY", "cuda:0")


def test_native_request_builds_cache_only_engine_input():
    _conversation, engine_inputs = OpenAIServingChat._build_cedfs_native_inputs(
        _request()
    )

    assert len(engine_inputs) == 1
    engine_input = engine_inputs[0]
    assert engine_input["prompt_token_ids"] == list(range(12))
    assert engine_input["mm_hashes"] == {"image": ["image-a"]}
    assert engine_input["mm_kwargs"]["image"] == [None]
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
        OpenAIServingChat._build_cedfs_native_inputs(
            _request([_feature(**{field: value})])
        )


def test_native_request_rejects_missing_server_context(monkeypatch):
    monkeypatch.delenv("CEDFS_OWNERSHIP_EPOCH")

    with pytest.raises(ValueError, match="context is not configured"):
        OpenAIServingChat._build_cedfs_native_inputs(_request())


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
    _conversation, engine_inputs = OpenAIServingChat._build_cedfs_native_inputs(
        request
    )

    assert engine_inputs[0]["mm_hashes"] == {"image": ["shared", "shared"]}


def test_native_request_rejects_overlapping_positions():
    with pytest.raises(ValueError, match="overlap"):
        OpenAIServingChat._build_cedfs_native_inputs(
            _request([_feature(position_offset=2), _feature(position_offset=4)])
        )
