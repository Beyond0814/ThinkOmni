#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

swift infer \
    --model model/model_stage3_SFA_AFA_MFR \
    --model_type qwen2_5_omni_xlsr \
    --template qwen2_5_omni_xlsr \
    --infer_backend pt \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --max_batch_size 8 \
    --dataset_num_proc 2 \
    --load_from_cache_file true \
    --val_dataset data/test_100k_linearspec.json \
    --max_new_tokens 4096 \
    --custom_register_path src/q25o_xlsr_cross_attn/custom_model.py \
    --temperature 0
