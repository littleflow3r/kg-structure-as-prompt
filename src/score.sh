#!/bin/bash

folder="checkpoints/comagc/"
cv="16-1,16-2,16-3,16-4,16-5"
tag="k0-n0-mistral-tca"
filescore="test_scores.txt"
avg="binary" #micro

python metric.py --folder ${folder} --cv ${cv} --tag ${tag} --filescore ${filescore} --avg ${avg}