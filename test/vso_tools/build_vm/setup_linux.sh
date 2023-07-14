#!/bin/bash

set -e
set -x

# Build essentials are required.
# But clean first...
sudo apt-get clean
sudo rm -rvf /var/lib/apt/lists/*
sudo apt-get clean
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo apt-get update
sudo apt-get install -y build-essential cmake uidmap

# Install azcli for Azure resources access and management.
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install azcopy for cache download.
# https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#use-azcopy-in-a-script
mkdir -p tmp
cd tmp
wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1
sudo cp ./azcopy /usr/bin/
sudo chmod +x /usr/bin/azcopy

# Install docker
# This docker must run with sudo.
# We don't know which user will run on pipeline in advance.
curl -fsSL https://get.docker.com | sh
sudo systemctl --now enable docker

# TODO: nvidia-docker should be installed here.

# Install NFS server / client
# This should only be done when neceessary, but it doesn't harm to install it, nonetheless.
# The NFS server can be accessed through the path: host.docker.internal
# Added a host alias so that it can also be used outside the container
# Inside the container they should use exactly the same uid/gid to read/write files.
sudo apt-get install -y nfs-kernel-server nfs-common
sudo mkdir -p /var/nfs/general
sudo chmod 777 /var/nfs/general
echo "/var/nfs/general *(rw,sync,insecure,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
echo "127.0.0.1 host.docker.internal" | sudo tee -a /etc/hosts
sudo systemctl restart nfs-kernel-server

# VM with GPU needs to install drivers. Reference:
# https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# https://linuxhint.com/install-cuda-ubuntu/
sudo apt-get install linux-headers-$(uname -r) -y
sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda-drivers

# UsePythonVersion task only works when the specific Python version is already installed.
# The following is for linux.
# Reference: https://dev.to/akaszynski/create-an-azure-self-hosted-agent-without-going-insane-173g
# We only need Python 3.7 and 3.9 for now.
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install -y python3.7-dev python3.7-venv python3.9-dev python3.9-venv python3.10-dev python3.10-venv python3.11-dev python3.11-venv

# Disable the periodical apt-get upgrade.
# Sometimes, unattended upgrade blocks apt-get install
sudo sed -i -e "s/Update-Package-Lists \"1\"/Update-Package-Lists \"0\"/g" /etc/apt/apt.conf.d/10periodic
sudo sed -i -e "s/Update-Package-Lists \"1\"/Update-Package-Lists \"0\"/g" /etc/apt/apt.conf.d/20auto-upgrades
sudo sed -i -e "s/Unattended-Upgrade \"1\"/Unattended-Upgrade \"0\"/g" /etc/apt/apt.conf.d/20auto-upgrades
sudo systemctl disable apt-daily.timer
sudo systemctl disable apt-daily.service
sudo systemctl disable apt-daily-upgrade.timer
sudo systemctl disable apt-daily-upgrade.service

# Deprovision
sudo /usr/sbin/waagent -force -deprovision
sudo HISTSIZE=0 sync