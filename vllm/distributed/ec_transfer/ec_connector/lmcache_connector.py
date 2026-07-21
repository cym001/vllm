# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LMCache EC Connector for vLLM v1.

This module bridges vLLM's ``ECConnectorBase`` interface to LMCache's
``LMCacheECConnectorImpl`` (defined in ``lmcache.integration.vllm.vllm_ec_adapter``).

Usage (via ``--ec-transfer-config``)::

    vllm serve <model> \
        --ec-transfer-config '{
          "ec_connector": "LMCacheECConnector",
          "ec_role": "ec_producer",
          "ec_connector_module_path":
              "vllm.distributed.ec_transfer.ec_connector.lmcache_connector"
        }'

Set ``LMCACHE_CONFIG_FILE`` in the environment to point to a YAML with at least
one storage backend configured for EC (``local_disk`` or ``local_cpu``).
"""

# Standard
from typing import TYPE_CHECKING, Any

# Third Party
import torch

# First Party
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheECConnector(ECConnectorBase):
    """vLLM-side EC connector that delegates to LMCache's ECCacheEngine.

    This class implements ``ECConnectorBase`` and wraps
    ``LMCacheECConnectorImpl`` from the ``lmcache`` package.
    The implementation lives in LMCache itself so that it can
    evolve independently of the vLLM release cycle.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        self._mm_datas_need_loads: dict[str, int] = {}

        # Lazy import: LMCache may not be installed in every deployment.
        from lmcache.integration.vllm.vllm_ec_adapter import (
            LMCacheECConnectorImpl,
        )

        self._impl = LMCacheECConnectorImpl(vllm_config, role, self)

    # ------------------------------------------------------------------
    # Worker-side methods (producer / consumer)
    # ------------------------------------------------------------------

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Load encoder caches from LMCache into vLLM's encoder_cache dict."""
        self._impl.start_load_caches(encoder_cache, **kwargs)

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        """Save one encoder cache entry into LMCache."""
        self._impl.save_caches(encoder_cache, mm_hash, **kwargs)

    # ------------------------------------------------------------------
    # Scheduler-side methods
    # ------------------------------------------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        """Check whether the EC cache for *identifier* exists in LMCache."""
        return self._impl.has_cache_item(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Track which mm item needs loading after scheduler allocation."""
        self._impl.update_state_after_alloc(request, index)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int] | None = None,
    ) -> tuple[bool, dict[str, Any] | None] | None:
        """Called when a request finishes; no-op for EC (cache is independent)."""
        return None

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Build metadata for the worker to know which caches to load."""
        return self._impl.build_connector_meta(scheduler_output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Return (finished_sending, finished_recving). No-op for EC."""
        return None, None

    def take_events(self) -> ECConnectorMetadata | None:
        """Return cached metadata (if any). EC does not batch events."""
        return None

    def close(self) -> None:
        """Release LMCache resources."""
        self._impl.close()
