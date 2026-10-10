# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalDataParser,
    Qwen2_5_VLProcessingInfo,
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


def test_video_embedding_parser_preserves_grid_timing():
    """v0.27's dictionary parser must not drop video time metadata."""
    info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2)
        ),
        _get_expected_hidden_size=lambda: 8,
    )
    parser = Qwen2_5_VLProcessingInfo.get_data_parser(info)
    assert isinstance(parser, Qwen2_5_VLMultiModalDataParser)
    data = {
        "video_embeds": torch.zeros(8, 8),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
        "second_per_grid_ts": torch.tensor([0.5]),
    }
    items = parser.parse_mm_data({"video": data})["video"]

    assert items.get_count() == 1
    assert items[0]["second_per_grid_ts"].item() == 0.5
    assert items.fields_config["second_per_grid_ts"].field.keep_on_cpu is True
