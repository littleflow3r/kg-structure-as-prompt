#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
tag_checkpoint='t5-fs-conv'
# tag_checkpoint='t5-full-conv'
model_name_or_config_path='t5-base'
mtask='comagc'

for k in "16-1" "16-2" "16-3" "16-4" "16-5"; do
    input_path="datasets/${mtask}/${k}"
    checkpoint_path="checkpoints/${mtask}/${k}/${tag_checkpoint}"
    python codes/conv_trainer.py \
    --input_path ${input_path} \
    --checkpoint_path ${checkpoint_path} \
    --model_name_or_config_path ${model_name_or_config_path} \
    --max_length 256 \
    --epoch 20 \
    --is_full 0 \
    --train_batch_size 8 \
    --eval_batch_size 1 \
    --lr 3e-5 \
    --warmup_proportion 0.06 \
    --save_result
done