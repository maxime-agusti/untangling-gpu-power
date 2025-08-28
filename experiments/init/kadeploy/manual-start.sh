#!/bin/bash

sudo systemctl stop dcgm

gpu_count=$(nvidia-smi --format=csv,noheader  --query-gpu=count | head -n 1)
for i in $(seq 0 $(($gpu_count-1)));
do
    sudo nvidia-smi -i $i -pm 1
    sudo nvidia-smi -i $i -mig 1
    sudo nvidia-smi -i $i --gpu-reset
done
sudo systemctl start dcgm

#docker run -d --gpus all --cap-add SYS_ADMIN --name dcgm-exporter --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:4.1.1-4.0.4-ubuntu22.04
docker run -d --gpus all --cap-add SYS_ADMIN --name dcgm-exporter --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:4.1.1-4.0.3-ubuntu22.04

(cd /home/pjacquet/gpu-burn && docker build . -t gpu_burn ) &
(cd init/bench/blender && docker build . -t blender) &
(cd init/bench/hpcg && docker build . -t hpcg) &
(cd init/bench/inference-llama && docker build . -t llama) &
(cd init/bench/training-yolo && docker build . -t yolo) &

sleep 5
while pgrep -f 'docker build' > /dev/null; do sleep 1; done