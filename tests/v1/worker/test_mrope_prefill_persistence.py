# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.mm.rope import RopeState


@pytest.mark.skipif(
    not torch.accelerator.is_available()
    or torch.accelerator.current_accelerator().type != "cuda",
    reason="requires CUDA UVA",
)
def test_recomputed_mrope_positions_persist_for_next_prefill_chunk():
    state = RopeState(
        num_dims=3,
        has_delta=True,
        max_num_reqs=1,
        max_num_tokens=16,
        max_model_len=16,
        device=torch.device("cuda"),
    )
    model = SimpleNamespace(
        get_mrope_input_positions=lambda _ids, _features: (
            torch.arange(12).repeat(3, 1),
            0,
        )
    )
    state.init_prefill_positions(0, model, list(range(12)), [], "request-1")
    state.apply_staged_writes()

    corrected = torch.arange(12, device="cuda").repeat(3, 1) + 7
    state.update_prefill_positions(0, corrected, -3, "request-1")
    torch.accelerator.synchronize()

    assert torch.equal(state.read_prefill_positions(0, 12).cpu(), corrected.cpu())
    state.init_prefill_positions(0, model, list(range(12)), [], "request-1")
    state.apply_staged_writes()
    assert torch.equal(state.read_prefill_positions(0, 12).cpu(), corrected.cpu())
    state.finish_request("request-1")
    assert "request-1" not in state.corrected_prefill


@pytest.mark.skipif(
    not torch.accelerator.is_available()
    or torch.accelerator.current_accelerator().type != "cuda",
    reason="requires CUDA UVA",
)
def test_streaming_update_with_same_length_does_not_restore_old_positions():
    state = RopeState(
        num_dims=3,
        has_delta=True,
        max_num_reqs=1,
        max_num_tokens=8,
        max_model_len=8,
        device=torch.device("cuda"),
    )
    model = SimpleNamespace(
        get_mrope_input_positions=lambda ids, _features: (
            torch.tensor(ids).repeat(3, 1),
            0,
        )
    )
    state.init_prefill_positions(0, model, [1, 2, 3], [], "request-1")
    state.apply_staged_writes()
    state.update_prefill_positions(
        0,
        torch.tensor([7, 8, 9], device="cuda").repeat(3, 1),
        -3,
        "request-1",
    )
    state.init_prefill_positions(0, model, [4, 5, 6], [], "request-1")
    state.apply_staged_writes()
    assert torch.equal(
        state.read_prefill_positions(0, 3).cpu(),
        torch.tensor([4, 5, 6]).repeat(3, 1),
    )
