#!/bin/bash

model='gpt-3.5-turbo-instruct'
mtask='semeval'

for k in "16-1" "16-4" "16-5" "16-6" "16-9"; do
    input_path="datasets/${mtask}/"
    checkpoint_path="checkpoints/${mtask}/"
    python src/fewshots.py \
    --input_path ${input_path} \
    --checkpoint_path ${checkpoint_path} \
    --model ${model} \
    --k ${k} \
    --n_samples 20 \
    --m_samples 2
done