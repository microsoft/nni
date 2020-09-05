#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# Initialize the user environment after starting MPS using init_mps_for_gpus.sh
# Supply the same device list as was given to init_mps_for_gpus.sh
# Run this script as a regular user sourcing environment settings: ". <script>.sh"
#
# set -x

# Get comma separated list of devices
devices=$1

# Message
echo "Setting MPS environment variables..."

# Create suffix for directory names by replacing all commas with underscores
suffix=`echo ${devices} | sed -e 's/,/_/g'`

# Set the environment variables for the daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps_$suffix
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps_$suffix

#set +x
