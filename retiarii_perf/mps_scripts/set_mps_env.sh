#!/bin/bash
# MPS Documentation: https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
#
# Initialize the user environment after starting MPS
# We look for the pipe directory in /tmp/nvidia-mps* and deduce which devices are in play.
#
#set -x

# Message
echo "Setting MPS environment variables..."

# Look in /tmp for a pipe directory
suffix=''
devices=''
pipedir=`ls /tmp | grep 'nvidia-mps'`
count=`echo $pipedir | wc -w`
if [ "$count" -gt '1' ]; then
  echo "There is more than one MPS environment defined."
  echo "Source set_mps_env_for_gpus.sh <gpu1>[,<gpu2>,...] instead"
else
  if [ "$pipedir" != '' ]; then
    suffix=`echo $pipedir | sed 's,nvidia-mps_\?,,'`
    devices=${suffix//_/,} # Substitute commas
  fi

  # Suffix is either a comma separated list of devices, or blank
  if [ "$devices" == '' ]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
  else
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps_$suffix
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps_$suffix
  fi
fi

#set +x
