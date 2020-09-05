#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# Initialize an mps daemon for a group of GPUs, supplied as a comma separated list on the command line
# This script can be used to set up multiple (disjoint) groups of CPUs, each group associated with one daemon
# Run the script as root.
# To stop the daemon associated with a group of GPUs, run stop_mps_for_gpus.sh with the same comma separated list
# There are three relevant environment variables:
# CUDA_VISIBLE_DEVICES which is a comma separated list of devices managed by the associated mps daemon
# CUDA_MPS_PIPE_DIRECTORY which is used to communicate with the daemon and server processes. This variable must
# also be set for client processes
# CUDA_MPS_LOG_DIRECTORY which is where logs are kept, and should also be set for client processes
# Note that we set our own values for these variables and we don't use the system defaults
# You need to set CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_PIPE_DIRECTORY for all client processes which are
# to use this group of GPUs

#set -x


# Get comma separated list of devices
devices=$1

# Message
echo "Start mps daemon for GPUs: $devices"

# Get total number of GPUs
NGPUS=`echo ${devices//,/ } | wc -w`

# Set the GPUs to exclusive process per Section 5.1.1.1
for i in ${devices//,/ }; do
  nvidia-smi -i $i -c EXCLUSIVE_PROCESS 1>/dev/null 
done

# Create suffix for directory names by replacing all commas with underscores
suffix=${devices//,/_}

# Set the environment variables for the daemon
export CUDA_VISIBLE_DEVICES=$devices 
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps_$suffix
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps_$suffix

# Create the pipe and log directories for the daemon
if [ ! -d $CUDA_MPS_PIPE_DIRECTORY ]; then
  mkdir $CUDA_MPS_PIPE_DIRECTORY
  chmod a+w $CUDA_MPS_PIPE_DIRECTORY
fi
if [ ! -d $CUDA_MPS_LOG_DIRECTORY ]; then
  mkdir $CUDA_MPS_LOG_DIRECTORY
  chmod a+w $CUDA_MPS_LOG_DIRECTORY
fi

# Run the daemon on CPU core #0, per Section 2.3.5.2
taskset -c 0 nvidia-cuda-mps-control -d 1>/dev/null 2>&1 

# Clean-up environment variables
unset CUDA_VISIBLE_DEVICES
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY

#set +x
