# CUDA_VISIBLE_DEVICES=5 swift infer \
#   --model_type minicpm-v-v2_6-chat \
#   --model /data/jj/ckpt/MiniCPM-V-2_6



  # 默认会将lora_target_modules设置为llm和resampler所有的linear
CUDA_VISIBLE_DEVICES=5  swift sft \
  --model_type minicpm-v-v2_6 \
  --model /data/jj/ckpt/MiniCPM-V-2_6 \
  --train_type lora \
  --dataset /data/jj/proj/MiniCPM-V/json_files/train_swift.json \
  --val_dataset /data/jj/proj/MiniCPM-V/json_files/val_swift.json \
  --deepspeed zero2 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_checkpointing \
  --max_seq_length 2048 \
  --max_train_steps 10000 \
  --use_flash_attn true

# Experimental environment: A100
# 80GB GPU memory
CUDA_VISIBLE_DEVICES=5  swift sft \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /data/jj/ckpt/MiniCPM-V-2_6 \
    --sft_type lora \
    --dataset /data/jj/proj/MiniCPM-V/json_files/colect50_endovis18_merged_train.json \
    --val_dataset /data/jj/proj/MiniCPM-V/json_files/colect50_val.json \
    --eval_steps 1000 \
    --output_dir output/output__lora \
    --logging_dir output/output_lora \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --num_train_epochs 3 \
    --max_length 2048 \
    --learning_rate 1e-6 \
    --use_flash_attn true \
    --save_only_model true \
    --preprocess_num_proc 4 \
    --batch_size 23 \
    --eval_batch_size 23 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to wandb


# get improved for cpu usage and training speed
CUDA_VISIBLE_DEVICES=5 swift sft \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /data/jj/ckpt/MiniCPM-V-2_6 \
    --sft_type lora \
    --dataset /data/jj/proj/MiniCPM-V/json_files/colect50_endovis18_merged_train.json \
    --val_dataset /data/jj/proj/MiniCPM-V/json_files/colect50_val.json \
    --eval_steps 1000 \
    --output_dir output/output__lora \
    --logging_dir output/output_lora \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --num_train_epochs 2 \
    --max_length 2048 \
    --learning_rate 3e-6 \
    --use_flash_attn true \
    --save_only_model true \
    --preprocess_num_proc 4 \
    --batch_size 28 \
    --eval_batch_size 28 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --warmup_steps 100 \
    --report_to wandb


# get improved for cpu usage and training speed
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type llama3_2-11b-vision-instruct \
    --model_id_or_path /data/jj/ckpt/Llama3.2-11B-Vision-Instruct \
    --sft_type lora \
    --dataset /data/jj/proj/MiniCPM-V/json_files/colect50_endovis18_merged_train.json \
    --val_dataset /data/jj/proj/MiniCPM-V/json_files/colect50_val.json \
    --eval_steps 1000 \
    --output_dir output/llama3_2_lora \
    --logging_dir output/llama3_2_lora \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --num_train_epochs 2 \
    --max_length 2048 \
    --learning_rate 3e-6 \
    --use_flash_attn true \
    --save_only_model true \
    --preprocess_num_proc 4 \
    --batch_size 32 \
    --eval_batch_size 32 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --warmup_steps 100 \
    --report_to wandb


CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type llama3-llava-next-8b-hf \
    --model_id_or_path swift/llama3-llava-next-8b-hf \
    --sft_type lora \
    --dataset /data/jj/proj/MiniCPM-V/json_files/colect50_endovis18_merged_train.json \
    --val_dataset /data/jj/proj/MiniCPM-V/json_files/colect50_val.json \
    --eval_steps 1000 \
    --output_dir output/llama3_2_lora \
    --logging_dir output/llama3_2_lora \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --num_train_epochs 2 \
    --max_length 2048 \
    --learning_rate 3e-6 \
    --use_flash_attn true \
    --save_only_model true \
    --preprocess_num_proc 4 \
    --batch_size 32 \
    --eval_batch_size 32 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --warmup_steps 100 \
    --report_to wandb

# inference
# Merge LoRA增量权重并推理
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=5 swift export \
    --ckpt_dir '/data/jj/proj/ms-swift/output/output__lora/minicpm-v-v2_6-chat/v1-20241206-205203/checkpoint-9498' --merge_lora true

CUDA_VISIBLE_DEVICES=5 swift infer \
--ckpt_dir '/data/jj/proj/ms-swift/output/output__lora/minicpm-v-v2_6-chat/v1-20241206-205203/checkpoint-9498-merged' --val_dataset '/data/jj/proj/MiniCPM-V/json_files/tmp.json'

CUDA_VISIBLE_DEVICES=5 swift infer \
--ckpt_dir '/data/jj/proj/ms-swift/output/output__lora/minicpm-v-v2_6-chat/v1-20241206-205203/checkpoint-9498' 

CUDA_VISIBLE_DEVICES=5 swift infer \
--ckpt_dir '/data/jj/proj/ms-swift/output/output__lora/minicpm-v-v2_6-chat/v1-20241206-205203/checkpoint-9498' --val_dataset '/data/jj/proj/MiniCPM-V/json_files/test_swiftjson' --use_flash_attn true

