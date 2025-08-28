# https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html
# https://docs.nvidia.com/datacenter/tesla/index.html
# kadeploy3 ubuntu2404-nfs
# oarsub -q testing -l walltime=2 -p chuc -t deploy -I
# kadeploy3 -a nvidia570.dsc

kadeploy3 ubuntu2404-nfs
oarsub -q testing -l walltime=2 -p chuc -t deploy -I

wget https://us.download.nvidia.com/tesla/570.124.06/nvidia-driver-local-repo-debian12-570.124.06_1.0-1_amd64.deb
dpkg -i nvidia-driver-local-repo-debian12-570.124.06_1.0-1_amd64.deb
cp /var/nvidia-driver-local-repo-debian12-570.124.06/nvidia-driver-local-BF654014-keyring.gpg /usr/share/keyrings/
apt update
apt install -y cuda-drivers # nvidia-open
#reboot 
sudo systemctl stop nvidia-persistenced
cat /proc/driver/nvidia/version

apt install -y ipmitool

# Install dcgm
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update && sudo apt-get install -y datacenter-gpu-manager
sudo systemctl enable dcgm 
sudo systemctl start dcgm 


# Install docker
sudo apt-get update
sudo apt-get -y install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo nano /etc/docker/daemon.json 
# {
#   "data-root": "/tmp/docker"
# }
mkdir /tmp/docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# Nvidia container runtime
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# g5k-postinstall --net debian --fstab nfs --restrict-user current

# tar xzf /grid5000/postinstalls/g5k-postinstall.tgz
# root@chuc-7:~# ./g5k-postinstall 