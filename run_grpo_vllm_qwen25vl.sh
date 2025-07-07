!/bin/bash
# Set thread configuration
export OMP_NUM_THREADS=1  # Prevent system overload
export CUDA_LAUNCH_BLOCKING=1

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./vllm_run.txt"

export WANDB_MODE="online"
export WANDB_API_KEY="your-wandb-key"

export WANDB_PROJECT="video-llm"  # name your W&B project
export WANDB_LOG_MODEL="false"  # log all model checkpoints
export WANDB_WATCH="all"


QWEN_PATH='Video-R1/Qwen2.5-VL-7B-COT-SFT'
HF_DATASET="video-r1-tutorial-data/NextQA_0_30_s_nextqa.json"
OUTPUT_DIR="./log/Qwen2.5-VL-7B-Video-GRPO-sft"
VIDEO_DIR="video-r1-tutorial-data/videos"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="Video-GRPO-no-temporal-with-length-test"
DS_CONFIG="scripts/zero3.json"  

# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
    --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
   open_r1/grpo.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --video_data_dir ${VIDEO_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 100352 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model false \
    --temporal true \
    --len_control true \
    --report_to wandb \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations 4 \
    --vllm_device "cuda:3" \
    --vllm_gpu_memory_utilization 0.7 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
