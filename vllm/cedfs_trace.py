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


def log_cedfs_ttft_event(logger: Logger, request_id: str, event: str) -> None:
    """Emit a machine-readable, host-monotonic CedFS TTFT event."""
    if envs.CEDFS_TRACE:
        logger.info(
            "CedFS TTFT trace: request_id=%s event=%s monotonic_ns=%d",
            request_id,
            event,
            time.monotonic_ns(),
        )
