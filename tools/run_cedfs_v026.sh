#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

vllm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(dirname "$vllm_root")"
vllm_python="${CEDFS_VLLM_PYTHON:-$vllm_root/.venv/bin/python}"
ec_python_root="${CEDFS_EC_PYTHON_ROOT:-$workspace_root/CedFs/cedfs-ec/python}"

if [[ ! -x "$vllm_python" ]]; then
    echo "Python environment not found: $vllm_python" >&2
    exit 1
fi

export PYTHONPATH="$vllm_root:$ec_python_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec "$vllm_python" -m vllm.entrypoints.cli.main "$@"
