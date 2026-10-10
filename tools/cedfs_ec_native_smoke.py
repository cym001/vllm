# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare video encoding with EC-native replay on a real GPU.

Uses ECExampleConnector and a fresh external cache key, not a CedFS server.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import safetensors.torch

from vllm import LLM, SamplingParams
from vllm.config import ECTransferConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


def generate(engine, request_id, inputs, params):
    engine.add_request(request_id, inputs, params)
    result = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                result = output
    if result is None:
        raise RuntimeError("No finished generation output")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()
    if args.num_frames < 2:
        parser.error("num-frames must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = args.output_dir / "ec"
    cache_dir.mkdir()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.3,
        limit_mm_per_prompt={
            "video": {"count": 1, "num_frames": 4, "width": 64, "height": 64}
        },
        video_pruning_rate=0.5,
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role="ec_both",
            ec_connector_extra_config={"shared_storage_path": str(cache_dir)},
        ),
    )
    frames = np.random.default_rng(0).integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    if args.video is not None:
        import cv2

        capture = cv2.VideoCapture(str(args.video))
        images = []
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError("Video must contain decodable frames")
            for index in np.linspace(0, count - 1, args.num_frames, dtype=int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, image = capture.read()
                if not success:
                    raise ValueError(f"Cannot decode video frame {index}")
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        frames = np.stack(images)
    prompt = (
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n"
        "Describe this video briefly.<|im_end|>\n<|im_start|>assistant\n"
    )
    engine = llm.llm_engine
    inputs = engine.input_processor.input_preprocessor.preprocess(
        {"prompt": prompt, "multi_modal_data": {"video": frames}}
    )
    params = SamplingParams(temperature=0, max_tokens=16)
    baseline = generate(engine, "baseline", inputs, params)
    source_key = inputs["mm_hashes"]["video"][0]
    source = cache_dir / source_key / "encoder_cache.safetensors"
    tensor = safetensors.torch.load_file(str(source))["ec_cache"]
    replay_key = "native-replay-" + source_key
    replay_dir = cache_dir / replay_key
    replay_dir.mkdir()
    shutil.copyfile(source, replay_dir / source.name)
    item = inputs["mm_kwargs"]["video"][0]
    position = inputs["mm_placeholders"]["video"][0]
    scope, fingerprint = "v026-smoke", "v026-smoke-bf16"
    os.environ.update(
        CEDFS_MODEL_SCOPES=scope,
        CEDFS_COMPATIBILITY_FINGERPRINT=fingerprint,
        CEDFS_OWNERSHIP_EPOCH="0",
        CEDFS_DEVICE_KEY="cuda:0",
    )
    request = ChatCompletionRequest(
        model=args.model,
        messages=[{"role": "user", "content": "native replay"}],
        cedfs_prompt_token_ids=inputs["prompt_token_ids"],
        cedfs_mm_features=[
            {
                "version": 1,
                "mm_hash": replay_key,
                "model_scope": scope,
                "num_encoder_tokens": tensor.shape[0],
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": "bfloat16",
                "compatibility_fingerprint": fingerprint,
                "ownership_epoch": 0,
                "device_key": "cuda:0",
                "modality": "video",
                "position_offset": position.offset,
                "position_length": position.length,
                "grid_thw": item["video_grid_thw"].data.tolist(),
                "second_per_grid_ts": item["second_per_grid_ts"].data.item(),
                "video_pruning_rate": 0.5,
            }
        ],
    )
    server = SimpleNamespace(model_config=engine.vllm_config.model_config)
    _, native_inputs = OpenAIServingChat._build_cedfs_native_inputs(server, request)
    cold = generate(engine, "native-cold", native_inputs[0], params)
    hot = generate(engine, "native-hot", native_inputs[0], params)
    outputs = [baseline, cold, hot]
    tokens = [list(output.outputs[0].token_ids) for output in outputs]
    result = {
        "model": args.model,
        "connector": "ECExampleConnector",
        "video": str(args.video) if args.video else "synthetic",
        "max_num_batched_tokens": 256,
        "q": 0.5,
        "shape": list(tensor.shape),
        "prompt_token_counts": [len(output.prompt_token_ids) for output in outputs],
        "generated_text": [output.outputs[0].text for output in outputs],
        "generated_token_ids": tokens,
        "equal": tokens[0] == tokens[1] == tokens[2],
        "external_replay_key": replay_key,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["equal"]:
        raise RuntimeError("Native EC cold/hot outputs differ from local encoding")


if __name__ == "__main__":
    main()
