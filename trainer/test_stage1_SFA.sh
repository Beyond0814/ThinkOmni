#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=0,1
swift infer \
    --model model/model_stage1_SFA \
    --model_type qwen2_5_omni \
    --template qwen2_5_omni \
    --merge_lora true \
    --infer_backend vllm \
    --val_dataset data/test_100k.json \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --temperature 0 
