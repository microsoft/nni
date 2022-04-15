#!/bin/bash
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact
pip install transformers==3.5.0

cur_dir=`pwd`
mkdir -p ${cur_dir}/log
arr=("jit" "tvm" "tvm-s" "trt" "rammer" "rammer-s" "sparta")
for framework in ${arr[@]}
do
    pushd $framework
    bash run.sh > ${cur_dir}/log/${framework}.log
    popd
done

python draw.py