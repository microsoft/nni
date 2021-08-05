#!/bin/bash -e

cd $(dirname $0)

export BACKEND=${BACKEND:-c-cuda}

if [[ "${BACKEND}" == "c-cuda" ]]; then
	export BACKEND="#cuda"
fi

if [[ "${BACKEND}" != "#cuda" ]]; then
	export LD_LIBRARY_PATH=/opt/tvm/build
else
	export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
fi

export HIP_PLATFORM=hcc
export HSA_USERPTR_FOR_PAGED_MEM=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/opt/tvm/python:/opt/tvm/topi/python:/opt/tvm/nnvm/python:/usr/local/rocm/src

ldconfig

time OP=${OP:-matmul} S=${S:-0} python3 ./compiler_auto_tune_stable.py "$@"

