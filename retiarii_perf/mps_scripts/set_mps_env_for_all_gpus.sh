#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# Set user environment variables for MPS after initializing daemon with init_mps_for_all_gpus.sh
# Run this script as a regular user sourcing a file for environment settings: ". <script>.sh"
#
# set -x

echo "Setting environment variables for MPS..."

# Default locations for the pipe and log directories
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

#set +x
