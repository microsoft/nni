#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# Stop an mps daemon (for a group of GPUs) which was started with init_mps_for_gpus.sh
# Supply the same comma separated list as used with init_mps_for_gpus.sh
# See init script for commentary about environment variables
#
#set -x

# Get comma separated list of devices
devices=$1

# Create suffix for directory names by replacing all commas with underscores
suffix=${devices//,/_}

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps_$suffix
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps_$suffix

if [ ! -d $CUDA_MPS_PIPE_DIRECTORY -o ! -d $CUDA_MPS_LOG_DIRECTORY ]; then
  echo "Cannot find either the pipe or log directory associated with supplied list."
  echo "Exiting"
  exit 1
fi

echo "Stopping mps daemon for GPUs: $devices"

# Stop the daemon
echo "quit" | nvidia-cuda-mps-control

#Remove the directories
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY

# Clean-up environment variables
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY

#set +x
