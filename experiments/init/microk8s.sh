sudo-g5k apt update
sudo-g5k apt install snapd
sudo-g5k snap install microk8s --classic

microk8s enable dns
microk8s enable hostpath-storage
microk8s enable gpu

microk8s kubectl logs -n gpu-operator-resources -lapp=nvidia-operator-validator -c nvidia-operator-validator

# sudo ufw allow in on cni0 && sudo ufw allow out on cni0
# sudo ufw default allow routed

microk8s kubectl get po -A

# Pistes


Changer version nvidia?
https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-570-86-15/index.html

Voir specificite A100
https://awslife.medium.com/how-to-install-nvidia-gpu-operator-with-a100-on-kubernetes-base-rocky-linux-2f888d476934

1.9.0 vs 24.x.x ??
annotation validation error: key "meta.helm.sh/release-name" must equal "gpu-operator-1741118562": current value is "gpu-operator-1741118512"