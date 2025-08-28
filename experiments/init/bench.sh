#!/bin/bash

g5k-setup-docker -t

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo-g5k gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo-g5k tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo-g5k apt-get update
sudo-g5k apt-get install -y nvidia-container-toolkit
sudo-g5k nvidia-ctk runtime configure --runtime=docker
sudo-g5k systemctl restart docker

(cd init/bench/blender && docker build . -t blender) &
(cd init/bench/hpcg && docker build . -t hpcg) &
(cd init/bench/inference-llama && docker build . -t llama) &
(cd init/bench/training-yolo && docker build . -t yolo) &

sleep 5
while pgrep -f 'docker build' > /dev/null; do sleep 1; done