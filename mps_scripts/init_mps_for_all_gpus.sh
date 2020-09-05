#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# This is the baseline script which will initialize **one** mps daemon for all GPUs
# Run this script as root.
# This uses the default pipe directory which is /tmp/nvidia-mps and default log directory /var/log/nvidia-mps
#
# To halt the mps daemon run stop_mps_for_all_gpus.sh (as root)
#
set -x

echo "Starting mps daemon for all GPUs..."

# Calculate number of GPUs
NGPUS=`nvidia-smi -L | wc -l`

# Create list for CUDA_VISIBLE_DEVICES
devices=0
for ((i=1;i<$NGPUS;i++))
{
  devices=${devices},$i
}

# This is used by the control daemon
export CUDA_VISIBLE_DEVICES=$devices

# Set the GPUs to exclusive mode per Section 5.1.1.1
for ((i=0;i<$NGPUS;i++))
{
  nvidia-smi -i $i -c EXCLUSIVE_PROCESS > /dev/null 2>&1 
}

# Start the control daemon on CPU core #0 per Section 2.3.5.2
ps -ef | grep -v grep | grep "nvidia-cuda-mps-control" 
if [ $? -ne 0 ]; then 
  taskset -c 0 nvidia-cuda-mps-control -d 
fi

set +x
