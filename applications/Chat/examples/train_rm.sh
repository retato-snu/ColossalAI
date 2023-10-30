#!/bin/bash

set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 2

torchrun --standalone --nproc_per_node=2 train_reward_model.py \
    --pretrain '/mnt/hf/polyglot-ko-1.3b' \
    --model 'polyglotko' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_exp' \
    --dataset 'json' \
    --data_path /mnt/ColossalAI/applications/Chat/examples/KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl \
    --batch_size 4 \
    --max_epochs 1 \
    --save_path output/polyglotko-1.3-rm \
    --data_bool False \
    --tokenizer '/mnt/hf/polyglot-ko-1.3b' \
    # --dataset 'Anthropic/hh-rlhf' \

