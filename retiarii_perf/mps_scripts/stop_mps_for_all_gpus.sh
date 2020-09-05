#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# This script will stop the mps daemon that was started with init_mps_for_all_gpus.sh
# Run this script as root
# There is an implicit assumption that the named pipe and log directories are in their default locations
# which are /tmp/mps_pipe and /tmp/mps_log
#set -x

echo "Killing mps daemon"
echo "quit" | nvidia-cuda-mps-control

rm -rf /tmp/nvidia-mps
rm -rf /var/log/nvidia-mps

#set +x
