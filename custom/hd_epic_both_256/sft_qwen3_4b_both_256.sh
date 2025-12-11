#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# DeepSpeed configuration
# deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=1
eval_ratio=0.0

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=hd_epic_both_256

# Output configuration
run_name="qwen3vl"
output_dir=./hd_epic_both_256

# Training arguments
args="
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 300000 \
    --min_pixels 5 \
    --eval_strategy no \
    --eval_ratio ${eval_ratio} \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    --report_to none"

# Launch training
# torchrun --nproc_per_node=${NPROC_PER_NODE} \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT} \
#          ${entry_file} ${args}

CUDA_VISIBLE_DEVICES=0 python ${entry_file} ${args}
