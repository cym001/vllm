# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)


def _feature(offset, length, grid):
    return SimpleNamespace(
        modality="video",
        mm_position=SimpleNamespace(offset=offset, length=length),
        data={
            "video_grid_thw": SimpleNamespace(
                data=SimpleNamespace(tolist=lambda: grid)
            ),
            "second_per_grid_ts": SimpleNamespace(
                data=SimpleNamespace(item=lambda: 0.5)
            ),
        },
    )


def test_pruned_multiple_videos_use_retained_placeholder_lengths():
    model = Qwen2_5_VLForConditionalGeneration.__new__(
        Qwen2_5_VLForConditionalGeneration
    )
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2)
    )
    features = [
        _feature(2, 4, [2, 4, 4]),  # 8 full-grid positions, 4 retained.
        _feature(9, 4, [2, 4, 4]),
    ]
    positions, delta = model.get_mrope_input_positions([0] * 15, features)
    assert positions.shape == (3, 15)
    assert positions[:, 6:9].shape == (3, 3)
    assert isinstance(delta, int)
