#!/bin/bash

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODEL_PATH="${MODEL_PATH:-model/model_xlsr_base}"
OUTPUT_DIR="${OUTPUT_DIR:-result/model_stage2_SFA+AFA}"
DATASET_PATH="${DATASET_PATH:-data/train_100k_think_linearspec.json}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-data/think_prompt_mel.text}"
CUSTOM_MODEL="${CUSTOM_MODEL:-src/q25o_xlsr_cross_attn/custom_model.py}"
CUSTOM_PLUGIN="${CUSTOM_PLUGIN:-src/q25o_xlsr_cross_attn/custom_plugin.py}"

# ===== Conda =====
# conda activate q25o xyx081430

NCCL_SHM_DISABLE=1 \
NCCL_CUMEM_HOST_ENABLE=0 \
MASTER_PORT=29800 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
        --external_plugins "$CUSTOM_MODEL" "$CUSTOM_PLUGIN" \
        --model "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset "$DATASET_PATH" \
        --model_type qwen2_5_omni_xlsr \
        --template qwen2_5_omni_xlsr \
        --load_from_cache_file true \
        --train_type custom \
        --optimizer custom \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --learning_rate 1e-4 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --attn_impl flash_attn \
        --padding_free true \
        --freeze_vit false \
        --vit_gradient_checkpointing true \
        --gradient_checkpointing true \
        --freeze_aligner false \
        --vit_lr 1e-5 \
        --aligner_lr 1e-5 \
        --dataloader_num_workers 4 \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 20 \
        --logging_steps 50 \
        --gradient_accumulation_steps 4 \
        --max_length 8192 \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --system "$SYSTEM_PROMPT" \
        --loss_type psd_detection_localization \
        --loss_scale psd_think_det_loc \
