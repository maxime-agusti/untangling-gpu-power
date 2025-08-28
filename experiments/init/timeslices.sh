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

sudo-g5k systemctl stop dcgm-exporter

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo-g5k dpkg -i minikube_latest_amd64.deb

minikube delete
minikube start --driver docker --container-runtime docker --gpus all
minikube kubectl -- get po -A

# curl -LO https://github.com/Mirantis/cri-dockerd/releases/download/v0.3.16/cri-dockerd_0.3.16.3-0.debian-bullseye_amd64.deb
# sudo-g5k dpkg -i cri-dockerd_0.3.16.3-0.debian-bullseye_amd64.deb

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash # Dangerous life
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update

#helm install --wait --generate-name -n gpu-operator --create-namespace nvidia/gpu-operator
helm install gpuo nvidia/gpu-operator --set driver.enabled=true --namespace kube-system --set mig.strategy=mixed --set devicePlugin.enabled=true --set operator.defaultRuntime=docker --set gfd.version=v0.8.1

eval $(minikube docker-env)
docker build -t gpu_burn /home/pjacquet/gpu-burn
#minikube image load gpu_burn

# minikube kubectl -- apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# minikube kubectl -- patch deployment metrics-server -n kube-system --type='json' -p='[
#   {"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"},
#   {"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-preferred-address-types=InternalIP"}
# ]'
#minikube kubectl -- delete pod -n kube-system -l k8s-app=metrics-server

POD_NAME=$(minikube kubectl -- get pods -n gpu-operator -l app=nvidia-dcgm-exporter -o jsonpath='{.items[0].metadata.name}')
minikube kubectl -- -n gpu-operator port-forward pod/$POD_NAME 9400:9400 > /dev/null 2>&1 &

# Debug
# minikube kubectl -- get clusterpolicies.nvidia.com/cluster-policy   -n gpu-operator -o yaml
# minikube kubectl -- get po -A
# minikube kubectl -- logs -n kube-system deployment/gpu-operator

# cat <<EOF | minikube kubectl -- create -n gpu-operator -f -
# apiVersion: v1
# kind: ConfigMap
# metadata:
#   name: oversub-all-2
# data:
#   any: |-
#     version: v1
#     flags:
#       migStrategy: none
#     sharing:
#       timeSlicing:
#         resources:
#         - name: nvidia.com/gpu
#           replicas: 2
# EOF

# minikube kubectl -- patch clusterpolicies.nvidia.com/cluster-policy \
#     -n gpu-operator --type merge \
#    -p '{"spec": {"devicePlugin": {"config": {"name": "oversub-all-2", "default": "any"}}}}'

# Test some images
# minikube kubectl create deployment hello-minikube --image=kicbase/echo-server:1.0
# cat <<EOF | minikube kubectl -- apply -f -
# apiVersion: v1
# kind: Pod
# metadata:
#   name: gpu-test
# spec:
#   restartPolicy: Never
#   containers:
#   - name: cuda-container
#     image: nvidia/cuda:11.8.0-runtime-ubuntu22.04
#     command: ["nvidia-smi"]
#     resources:
#       limits:
#         nvidia.com/gpu: 1
# EOF

# cat <<EOF | minikube kubectl -- apply -f -
# apiVersion: v1
# kind: Pod
# metadata:
#   name: gpu-burn
# spec:
#   restartPolicy: Never
#   containers:
#   - name: container-burn
#     image: gpu_burn
#     imagePullPolicy: Never
#     command: ["./gpu_burn","-d","120"]
#     resources:
#       limits:
#         nvidia.com/gpu: 1
# EOF
#minikube kubectl -- delete pod gpu-burn