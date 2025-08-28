#!/bin/bash

git clone https://github.com/wilicc/gpu-burn
cd gpu-burn

ARG CUDA_VERSION=11.8.0                                                                                   │
ARG IMAGE_DISTRO=ubuntu22.04
docker build -t gpu_burn .

gpu_count=$(nvidia-smi --format=csv,noheader  --query-gpu=count | head -n 1)
for i in $(seq 0 $(($gpu_count-1)));
do
    sudo nvidia-smi -i $i -pm 1
    sudo nvidia-smi -i $i -mig 1
    sudo nvidia-smi -i $i --gpu-reset
done