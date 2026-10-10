# CedFS-EC / EVS v0.26.0 回迁与部署

适用日期：2026-10-10。分支为 `feat/ced-ec-evs-v0.26.0`，基于官方 `v0.26.0`（`568afb3a13806beb53bb2e6bd518269357b237c0`），从 v0.27.1 集成提交 `a897aa01d1` 回迁 Python 补丁。仅支持传统 Model Runner；不引入 V2 或旧分支 P0 的 Marlin 生成内核改动。

## 改动边界

- 保留 v0.26 的 `vllm.multimodal.evs` 路径，将视频位置修复直接应用于旧实现。
- 为 `MultiModalConfig` 增加 `get_video_pruning_spec()`，只返回 EVS 的 `(method, rate)`，不引入 VidCom2 或新剪枝配置。
- 回迁异步 EC 三态、严格 consumer、零 token 轮询、完成回传、请求级失败和取消清理。
- 回迁视频 EC-native v2/v3、M2 内容采样与 variant 校验、时间/网格透传、多视频及分块 M-RoPE、剪枝视频 KV 前缀读取防护。
- 保留图像原生请求、route token、外部缓存元数据和可选 TTFT 追踪；配套测试适配 v0.26 的缓存接口。
- 原始依赖清单和 C++/CUDA 源码保持 v0.26。M2 仍禁用在线异步 EC 预取，raw BF16 是默认部署 codec；引擎的通用异步 EC 能力不代表 M2 已开启该路径。

## 复用已有环境

本机 `.venv` 链接到 `/home/chenym/cedfs-ec/.venv`，复用 Python 3.10.12、PyTorch 2.11.0+cu130 和第三方依赖。没有改写原环境的 vLLM editable 安装。新工作树使用匹配的官方 v0.26 wheel 原生扩展及 FlashAttention 附带文件；这些是本地、Git 忽略的产物，不包含在源码提交里。

使用启动包装器选择新源码，避免直接调用共享环境的 `vllm` 入口而误入旧工作树：

```bash
cd /home/chenym/cedfs-ec/vllm-v0.26.0-cedfs-ec
bash tools/run_cedfs_v026.sh serve /path/to/model --help
```

包装器设置新工作树和 CedFS Python 路径，固定 `VLLM_USE_V2_MODEL_RUNNER=0`、`spawn`。可用 `CEDFS_VLLM_PYTHON` 指定另一 Python，用 `CEDFS_EC_PYTHON_ROOT` 指定另一 CedFS 包路径。共享环境的发行版 metadata 仍可能显示旧 editable 构建标签；确认 `vllm.__file__` 和 `vllm.__version__`，不要仅按 console 的 metadata 标签判断导入来源。

```bash
export PYTHONPATH=/home/chenym/cedfs-ec/vllm-v0.26.0-cedfs-ec:/home/chenym/cedfs-ec/CedFs/cedfs-ec/python
.venv/bin/python -c 'import vllm, torch; print(vllm.__file__, vllm.__version__); print(torch.__version__, torch.version.cuda)'
```

在另一台主机安装时，使用该主机允许的 v0.26 构建、PyTorch/CUDA 和 GPU 架构，不复制这些绝对路径链接。重建/安装 v0.26 的预编译产物或源码构建，再安装 CedFS `_native` 扩展；不能混用 v0.27/v0.29 扩展。示例配置中的模型、媒体和 CedFS TOML 路径也必须同步调整。

## EVS + CedFS 启动

配置位于 [cedfs_v026](../examples/disaggregated/disaggregated_encoder/cedfs_v026/)，包含环境模板、Producer 和 PD 的 EC JSON。每个终端先 source 环境：

```bash
source /home/chenym/cedfs-ec/vllm-v0.26.0-cedfs-ec/examples/disaggregated/disaggregated_encoder/cedfs_v026/deployment.env.example
```

使用已有健康的 CedFS Meta/Data/Cache 服务；配置中的 client TOML 当前指向 `10.70.251.47:8001`。不要用会传入 `--need-reset-storage` 的测试启动脚本复用持久化数据。新机器同步修改 client/data/meta 的地址、路由、TCP/UDP 端口和存储根；同机共享内存不能代替跨机传输。

Producer（独立终端）：

```bash
cd "$VLLM_ROOT"
CUDA_VISIBLE_DEVICES="$PRODUCER_GPU" bash tools/run_cedfs_v026.sh serve "$MODEL" \
  --host 127.0.0.1 --port 18090 --dtype bfloat16 --max-model-len 4096 \
  --seed 0 --video-pruning-rate 0.5 --limit-mm-per-prompt '{"video":2}' \
  --allowed-local-media-path "$MEDIA_ROOT" \
  --ec-transfer-config "$(jq -c . "$DEPLOY_ROOT/producer-ec.json")"
```

PD（独立终端）：

```bash
cd "$VLLM_ROOT"
CUDA_VISIBLE_DEVICES="$PD_GPU" bash tools/run_cedfs_v026.sh serve "$MODEL" \
  --host 127.0.0.1 --port 18091 --dtype bfloat16 --max-model-len 4096 \
  --seed 0 --video-pruning-rate 0.5 --limit-mm-per-prompt '{"video":2}' \
  --ec-transfer-config "$(jq -c . "$DEPLOY_ROOT/pd-ec.json")"
```

Proxy（独立终端）：

```bash
cd "$EC_ROOT"
"$VLLM_ROOT/.venv/bin/python" -m cedfs_ec.epd_proxy \
  --host 127.0.0.1 --port 18092 \
  --encode-servers-urls "$PRODUCER_URL" \
  --prefill-servers-urls disable --decode-servers-urls "$PD_URL" \
  --encoder-dispatch-policy lookup-first --cedfs-config "$CEDFS_CONFIG" \
  --candidate-scopes "$CEDFS_MODEL_SCOPES" \
  --pd-request-format ec-native --native-model-scope "$CEDFS_MODEL_SCOPES" \
  --video-pruning-rate 0.5 --evs-m2 --m2-video-fps 0.5 \
  --preprocessor-config-sha256 "$PREPROCESSOR_SHA256" \
  --m2-codec evs-raw-bf16 --m2-calibration-sha256 "$CODEC_CONFIG_SHA256" \
  --spatial-merge-size 2 --model-revision "$CEDFS_MODEL_REVISION" \
  --processor-revision "$CEDFS_PROCESSOR_REVISION" --evs-version "$CEDFS_EVS_VERSION" \
  --compatibility-fingerprint "$CEDFS_COMPATIBILITY_FINGERPRINT" \
  --ownership-epoch "$CEDFS_OWNERSHIP_EPOCH" --device-key "$CEDFS_DEVICE_KEY"
```

JSON 基础 scope 为 `evsv0260_raw_r1`，真实 scope 自动追加 `_video_evs_q0p5`；不要直接使用 v0.27 或 v0.29 的 scope/revision。`CUDA_VISIBLE_DEVICES=5` 下 PD 的本地设备仍是 `cuda:0`。PD 业务严格消费 EC，但上述命令没有移除视觉权重；language-only 显存优化、V2、多机分离 P/D 和实验量化不属于本次验证范围。

三端 `GET /health` 正常后，经 Proxy `/v1/chat/completions` 发送 `video_url` + 文本请求。M2 使用三端可读的同源本地路径、`file://` 或 `data:`，固定 fps=0.5；`--evs-m2` 与 `--evs-v2` 互斥。检查 COMPLETE 对象、K/shape/pos4、冷/热命中与 Producer 编码次数，而不仅是 HTTP 200。

## 验证与边界

2026-10-10 已完成的验证：

- vLLM 相关回归：483 passed，2 项需额外模型权重的上游测试未运行。
- CedFS 侧相关回归：139 passed。
- GPU 合成视频：本地编码、换新 key 的 EC 冷加载和热请求的 16 个生成 token 全部一致。
- GPU 真实 MVBench `RPY8D.mp4`，均匀采样 8 帧、q=0.5：EC 为 `494×2052` BF16，三个请求均为 510 个 prompt token，16 个生成 token 全部一致。`max_num_batched_tokens=256`，实际覆盖分块 prefill。

原始结果保存在本机 `/home/chenym/tmp/vllm-v026-precompiled.cDdovC/` 下的 `vllm-regression.xml`、`native-smoke/result.json` 和 `real-video-smoke/result.json`。这个目录同时保存新工作树链接使用的官方 v0.26 原生扩展，复用环境期间不要删除；新机器应重新安装匹配产物。

回归命令：

```bash
cd "$VLLM_ROOT"
.venv/bin/python -m pytest -q \
  tests/config/test_multimodal_config.py \
  tests/entrypoints/openai/test_cedfs_native_request.py \
  tests/model_executor/test_evs_multi_video_mrope.py \
  tests/model_executor/test_qwen2_5_vl_pruned_mrope.py \
  tests/model_executor/test_qwen3_vl_mrope.py tests/test_cedfs_trace.py \
  tests/v1/worker/test_gpu_model_runner_mm_gather.py \
  tests/v1/worker/test_mrope_prompt_embeds.py \
  tests/v1/core/test_scheduler.py tests/multimodal/test_cache.py \
  tests/test_envs.py tests/v1/ec_connector/unit \
  -k 'not sleep_wake_preserves_mm_cache_consistency and not ec_cpu_offloading'
```

排除的两项上游测试需要另一个完整模型 Qwen2-VL-2B 的权重；这不是跳过本次 native/EVS 回归。CedFS 侧已用共享环境及新源码路径运行 cross-domain、connector、proxy、EVS、native、M2 codec、native request preparation 和 calibration 相关测试。

GPU 冒烟工具使用独立 ECExampleConnector 文件缓存，对拍实际本地视频编码、换新缓存键的外部 EC 冷请求及热请求：

```bash
cd "$VLLM_ROOT"
CUDA_VISIBLE_DEVICES=4 CEDFS_M1_VERIFY_POSITIONS=1 \
.venv/bin/python tools/cedfs_ec_native_smoke.py \
  --model "$MODEL" \
  --video "$MEDIA_ROOT/video/sta/sta_video/RPY8D.mp4" --num-frames 8 \
  --output-dir /path/to/new-smoke-directory
```

输出目录必须不存在。此工具验证实际 vLLM GPU / 视频 / EC-native 数据路径，不能等同真实 CedFS 存储闭环，也不能替代完整数据集质量评估。目标机器上线前还需在独立 scope 下跑真实 CedFS 冷/热、双视频、缺失/损坏对象、取消和并发恢复，并保存原生/EC 的 token、位置和质量对拍结果。

硬件变化时重新验证 GPU 架构、驱动/runtime、attention 后端、显存和并发预算，以及媒体采样/Processor/EVS 输出。缓存身份不会自动包含所有硬件信息；新 Producer 首先使用隔离 scope 和新修订标签，确认兼容后再决定旧对象能否复用。monotonic 日志不能直接跨主机相减。保留旧分支、环境和对象作为回滚基线。
