# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.multimodal.video_prune.evs import (
    compute_retention_mask,
    recompute_mrope_positions,
)


def test_multiple_videos_have_identical_mrope_with_chunked_prefill():
    # text, vision_start, video x4, text, vision_start, video x4, text
    input_ids = torch.tensor([1, 9, 7, 7, 7, 7, 2, 9, 7, 7, 7, 7, 3])
    initial = torch.arange(len(input_ids)).repeat(3, 1)
    media = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [2, 2, 2, 2],
        ]
    )

    def recompute(items, positions, computed):
        return recompute_mrope_positions(
            input_ids,
            items,
            positions,
            computed,
            vision_start_token_id=9,
            image_token_id=6,
            video_token_id=7,
        )

    whole, whole_delta = recompute([media, media], initial, 0)
    first, _ = recompute([media], initial, 0)
    chunked, chunked_delta = recompute([media], first, 8)

    assert torch.equal(whole, chunked)
    assert whole_delta == chunked_delta


def test_zero_retained_intermediate_video_frame():
    # The first frame is mandatory; only frame 1 differs. Stable top-k then
    # removes every token from frames 2 and 3, including an intermediate frame.
    first = torch.tensor([1.0, 0.0]).repeat(4, 1)
    changed = torch.tensor([0.0, 1.0]).repeat(4, 1)
    embeddings = torch.cat([first, changed, changed, changed])
    mask = compute_retention_mask(embeddings, (4, 2, 2), 1, 0.5)
    assert mask.reshape(4, 4).sum(dim=1).tolist() == [4, 4, 0, 0]
