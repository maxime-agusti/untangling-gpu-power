#!/bin/bash

echo "Deploying: $OAR_JOB_ID"
kadeploy3 -a /home/pjacquet/kadeploy-nvidia570.yml
host=$( oarstat -j $OAR_JOB_ID -f | grep 'assigned_hostnames' | cut -d '=' -f2 | cut -c2- | cut -d '.' -f1 )
if [ "$(echo "$host" | wc -l)" -eq 1 ]; then
  echo "Host retrieved: $host > Proceeding"
  
  ssh pjacquet@$host << 'EOF'
    gpu_count=$(nvidia-smi --format=csv,noheader --query-gpu=count | head -n 1)
    for i in $(seq 0 $(($gpu_count-1))); do
      sudo nvidia-smi -i $i -pm 1
      sudo nvidia-smi -i $i -mig 1
      sudo nvidia-smi -i $i --gpu-reset
    done
    sudo reboot
EOF

  echo "Waiting for server to come back online..."
  sleep 2
  
  while ! ssh -o ConnectTimeout=5 pjacquet@$host "echo 'Server is back up'"; do 
    sleep 10
  done

  echo "Server is back up, running Docker commands..."

  ssh pjacquet@$host << 'EOF'
    gpu_count=$(nvidia-smi --format=csv,noheader --query-gpu=count | head -n 1)
    for i in $(seq 0 $(($gpu_count-1))); do
      sudo nvidia-smi -i $i -mig 1
    done
    set -e  # Exit on error

    docker run -d --gpus all --cap-add SYS_ADMIN --name dcgm-exporter --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:4.1.1-4.0.3-ubuntu22.04

    (cd /home/pjacquet/gpu-burn && docker build . -t gpu_burn) &
    (cd /home/pjacquet/cloud-gpu-manager/init/bench/blender && docker build . -t blender) &
    (cd /home/pjacquet/cloud-gpu-manager/init/bench/hpcg && docker build . -t hpcg) &
    (cd /home/pjacquet/cloud-gpu-manager/init/bench/inference-llama && docker build . -t llama) &
    (cd /home/pjacquet/cloud-gpu-manager/init/bench/training-yolo && docker build . -t yolo) &

    sleep 5
    while pgrep -f 'docker build' > /dev/null; do sleep 1; done
EOF

  echo "End of setup, launching"

  ssh pjacquet@$host << 'EOF'
    cd /home/pjacquet/cloud-gpu-manager && python3 exp-perf-mig-ci.py
EOF

  echo "End of experiment, exit"
  #sleep infinity
else
  echo "Error condition: multiple hosts detected, will sleep indefinitely."
  sleep infinity
fi