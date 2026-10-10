# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.cedfs_trace import (
    TRACE_SCHEMA_VERSION,
    get_cedfs_request_id,
    log_cedfs_ttft_event,
)
from vllm.v1.engine.input_processor import InputProcessor


def test_get_cedfs_request_id_is_explicit():
    assert get_cedfs_request_id({"X-CedFS-Request-Id": "bench-7"}) == "bench-7"
    assert get_cedfs_request_id({"X-Request-Id": "bench-7"}) is None
    assert get_cedfs_request_id(None) is None


def test_cedfs_ttft_trace_defaults_off(monkeypatch):
    monkeypatch.delenv("CEDFS_TRACE", raising=False)
    logger = Mock()

    log_cedfs_ttft_event(logger, "bench-7", "engine_enqueue")

    logger.info.assert_not_called()


def test_cedfs_ttft_trace_uses_monotonic_clock(monkeypatch):
    monkeypatch.setenv("CEDFS_TRACE", "1")
    logger = Mock()

    log_cedfs_ttft_event(logger, "bench-7", "engine_enqueue")

    args = logger.info.call_args.args
    assert args[:3] == (
        (
            "CedFS TTFT trace: request_id=%s event=%s monotonic_ns=%d "
            "trace_schema_version=%s"
        ),
        "bench-7",
        "engine_enqueue",
    )
    assert isinstance(args[3], int)
    assert args[3] > 0
    assert args[4] == TRACE_SCHEMA_VERSION


def test_cedfs_trace_preserves_external_request_id(monkeypatch):
    monkeypatch.setenv("CEDFS_TRACE", "1")
    request = SimpleNamespace(request_id="bench-7", external_req_id=None)

    InputProcessor.assign_request_id(request)

    assert request.request_id == "bench-7"
    assert request.external_req_id == "bench-7"


def test_cedfs_trace_accepts_hot_path_phase_events(monkeypatch):
    monkeypatch.setenv("CEDFS_TRACE", "1")
    logger = Mock()

    for event in (
        "pd_api_parsed",
        "mm_processor_done",
        "encoder_cache_attached",
        "embedding_merge_done",
    ):
        log_cedfs_ttft_event(logger, "bench-7", event)

    assert [call.args[2] for call in logger.info.call_args_list] == [
        "pd_api_parsed",
        "mm_processor_done",
        "encoder_cache_attached",
        "embedding_merge_done",
    ]
