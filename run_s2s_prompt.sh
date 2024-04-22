#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
tag_checkpoint='t5-pr2-mp'
model_name_or_config_path='t5-base'
mtask='comagc'

# for k in "16-1" "16-2" "16-3" "16-4" "16-5"; do
for k in "mp-1" "mp-2" "mp-3" "mp-4" "mp-5"; do #metapath
# for k in "nn-1" "nn-2" "nn-3" "nn-4" "nn-5"; do #neighbors nodes
# for k in "cnein-1" "cnein-2" "cnein-3" "cnein-4" "cnein-5"; do #common neighbors nodes
    input_path="datasets/${mtask}/${k}"
    checkpoint_path="checkpoints/${mtask}/${k}/${tag_checkpoint}"
    python src/prompt_trainer_t5.py \
    --input_path ${input_path} \
    --checkpoint_path ${checkpoint_path} \
    --model_name_or_config_path ${model_name_or_config_path} \
    --max_length 256 \
    --with_evs 1 \
    --prompt_text "The pair marked with <e1> and <e2> shows a causal relation : " \
    --train_batch_size 8 \
    --eval_batch_size 1 \
    --epoch 20 \
    --lr 3e-5 \
    --warmup_proportion 0.06 \
    --save_result
done
