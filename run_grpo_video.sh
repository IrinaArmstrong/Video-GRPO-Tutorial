!/bin/bash
# Set thread configuration
export OMP_NUM_THREADS=1  # Prevent system overload
export CUDA_LAUNCH_BLOCKING=1

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

export WANDB_MODE="online"
export WANDB_API_KEY="your-wandb-key"

export WANDB_PROJECT="video-grpo-tutorial"  # name your W&B project
export WANDB_LOG_MODEL="false"  # log all model checkpoints
export WANDB_WATCH="all"

VIDEO_DIR="video-r1-tutorial-data/videos"

# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    open_r1/grpo.py \
    --output_dir "./log/Qwen2.5-VL-7B-GRPO" \
    --model_name_or_path 'Video-R1/Qwen2.5-VL-7B-COT-SFT' \
    --video_data_dir ${VIDEO_DIR} \
    --dataset_name "/video-r1-tutorial-data/NextQA_0_30_s_nextqa.json" \
     --deepspeed scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing false \
    --temporal false \
    --len_control false \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 100352 \
    --num_train_epochs 1 \
    --run_name Video-GRPO-with-temporal-with-length \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 4  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance     