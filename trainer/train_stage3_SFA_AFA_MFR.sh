#!/bin/bash

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODEL_PATH="${MODEL_PATH:-model/model_stage2_SFA+AFA}"
OUTPUT_DIR="${OUTPUT_DIR:-result/model_stage3_SFA+AFA+MFR}"
DATASET_PATH="${DATASET_PATH:-data/train_100k_think_linearspec.json}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-data/think_prompt_mel.text}"
CUSTOM_MODEL="${CUSTOM_MODEL:-src/q25o_xlsr_cross_attn/custom_model.py}"

# ===== Conda =====
# conda activate q25o xyx081430
# --gradient_checkpointing_kwargs '{"use_reentrant": false}' \

NCCL_SHM_DISABLE=1 \
NCCL_CUMEM_HOST_ENABLE=0 \
MASTER_PORT=29800 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
        --external_plugins "$CUSTOM_MODEL" \
        --model "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset "$DATASET_PATH" \
        --model_type qwen2_5_omni_xlsr \
        --template qwen2_5_omni_xlsr \
        --load_from_cache_file true \
        --train_type lora \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-4 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --freeze_vit false \
        --freeze_llm false \
        --freeze_aligner false \
        --padding_free true \
        --attn_impl flash_attn \
        --vit_lr 1e-5 \
        --aligner_lr 1e-5 \
        --dataloader_num_workers 4 \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 50 \
        --logging_steps 100 \
        --gradient_accumulation_steps 4 \
        --max_length 8192 \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --loss_type psd_detection_localization \
        --loss_scale psd_think_det_loc \
        --system "$SYSTEM_PROMPT" \
