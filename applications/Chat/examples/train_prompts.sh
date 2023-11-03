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

# torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy colossalai_zero2

torchrun --standalone --nproc_per_node=2 train_prompts.py \
    --model polyglotko \
    --pretrain output/polyglot-ko-12.8b-lora-koChat \
    --pretrain_dataset /mnt/ColossalAI/applications/Chat/examples/KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl  \
    --prompt_dataset /mnt/ColossalAI/applications/Chat/examples/KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl \
    --strategy colossalai_zero2 \
    --rm_model 'gpt-neox' \
    --rm_pretrain "/mnt/hf/korean-gpt-neox-125M" \
    --rm_path output/korean-gpt-neox-125M-rm \
    --num_episodes 1 --num_collect_steps 2 --num_update_steps 1 \
    --train_batch_size 4 \
    --save_path output/ppo_polyglotko-12.8+neox125_test \
    --lora_rank 8 \
    --language 'ko' \
    --instruction_str "prompt"\
    --output_str "completion" \
    --ppo_instruction_str "prompt"