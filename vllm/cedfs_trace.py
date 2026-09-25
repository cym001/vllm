# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from logging import Logger

import vllm.envs as envs


def get_cedfs_request_id(headers: object | None) -> str | None:
    """Return the explicit cross-process request ID, when provided."""
    if headers is None or not hasattr(headers, "get"):
        return None
    request_id = headers.get("X-CedFS-Request-Id")  # type: ignore[union-attr]
    return str(request_id) if request_id else None


TRACE_SCHEMA_VERSION = "cedfs-ec-trace-v2"


def log_cedfs_ttft_event(logger: Logger, request_id: str, event: str) -> None:
    """Emit a machine-readable, host-monotonic CedFS TTFT event.

    v2 distinguishes scheduler readiness from a physical CPU-to-GPU copy.
    The version is emitted on every record so mixed historical logs fail
    closed in analysis instead of silently acquiring current semantics.
    """
    if envs.CEDFS_TRACE:
        logger.info(
            "CedFS TTFT trace: request_id=%s event=%s monotonic_ns=%d trace_schema_version=%s",
            request_id,
            event,
            time.monotonic_ns(),
            TRACE_SCHEMA_VERSION,
        )
