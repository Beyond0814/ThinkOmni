#!/bin/bash

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODEL_PATH="${MODEL_PATH:-model/Qwen2.5-Omni-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-result/Q25O7B_100k_Think_Stage1SFA}"
DATASET_PATH="${DATASET_PATH:-data/train_100k_think.json}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-data/think_prompt.text}"

NCCL_SHM_DISABLE=1 \
NCCL_CUMEM_HOST_ENABLE=0 \
MASTER_PORT=29900 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
        --model "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset "$DATASET_PATH" \
        --model_type qwen2_5_omni \
        --load_from_cache_file true \
        --train_type lora \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-4 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --attn_impl flash_attn \
        --padding_free true \
        --freeze_vit false \
        --freeze_llm false \
        --vit_gradient_checkpointing true \
        --gradient_checkpointing true \
        --freeze_aligner false \
        --vit_lr 1e-5 \
        --aligner_lr 1e-5 \
        --dataloader_num_workers 4 \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 10 \
        --logging_steps 100 \
        --gradient_accumulation_steps 4 \
        --max_length 8192 \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --system "$SYSTEM_PROMPT" \
        --loss_type psd_detection_localization \
        --loss_scale psd_think_det_loc \
