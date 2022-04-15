# measure the jit time
cur_dir=`pwd`
source ~/.bashrc
echo "Curret directory ${cur_dir}"
mkdir ${cur_dir}/log

source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact


########################JIT###############################
pip install transformers==3.5.0
# bert coarse jit
pushd bert_coarse_jit
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_jit.log
popd

pushd bert_finegrained_jit
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_jit.log
popd

pushd bert_coarse_int8_jit
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_jit.log
popd

pushd mobilenet_coarse_jit
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_jit.log
popd

pushd mobilenet_finegrained_jit
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_jit.log
popd

pushd mobilenet_coarse_int8_jit
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_jit.log
popd
# change the python environment for hubert (which is only available in transformers 4.12)

pip install transformers==4.12.3
pushd hubert_coarse_jit
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_jit.log
popd

pushd hubert_finegrained_jit
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_jit.log
popd

pushd hubert_coarse_int8_jit
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_jit.log
popd

########################TensorRT###########################
pushd bert_coarse_trt
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_trt.log
popd

pushd bert_coarse_int8_trt
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_trt.log
popd

pushd bert_finegrained_trt
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_trt.log
popd


pushd mobilenet_coarse_trt
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_trt.log
popd

pushd mobilenet_coarse_int8_trt
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_trt.log
popd

pushd mobilenet_finegrained_trt
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_trt.log
popd

pushd hubert_coarse_trt
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_trt.log
popd

pushd hubert_coarse_int8_trt
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_trt.log
popd

pushd hubert_finegrained_trt
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_trt.log
popd

########################TVM###########################
pushd bert_coarse_tvm
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_tvm.log
popd

pushd bert_coarse_int8_tvm
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_tvm.log
popd

pushd bert_finegrained_tvm
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_tvm.log
popd


pushd mobilenet_coarse_tvm
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_tvm.log
popd

pushd mobilenet_coarse_int8_tvm
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_tvm.log
popd

pushd mobilenet_finegrained_tvm
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_tvm.log
popd

pushd hubert_coarse_tvm
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_tvm.log
popd

pushd hubert_coarse_int8_tvm
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_tvm.log
popd

pushd hubert_finegrained_tvm
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_tvm.log
popd

########################TVM-SPARSE###########################
pushd bert_coarse_tvm-s
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_tvm-s.log
popd

pushd bert_coarse_int8_tvm-s
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_tvm-s.log
popd

pushd bert_finegrained_tvm-s
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_tvm-s.log
popd


pushd mobilenet_coarse_tvm-s
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_tvm-s.log
popd

pushd mobilenet_coarse_int8_tvm-s
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_tvm-s.log
popd

pushd mobilenet_finegrained_tvm-s
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_tvm-s.log
popd

pushd hubert_coarse_tvm-s
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_tvm-s.log
popd

pushd hubert_coarse_int8_tvm-s
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_tvm-s.log
popd

pushd hubert_finegrained_tvm-s
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_tvm-s.log
popd

########################RAMMER###########################
pushd bert_coarse_rammer
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_rammer.log
popd

pushd bert_coarse_int8_rammer
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_rammer.log
popd

pushd bert_finegrained_rammer
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_rammer.log
popd


pushd mobilenet_coarse_rammer
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_rammer.log
popd

pushd mobilenet_coarse_int8_rammer
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_rammer.log
popd

pushd mobilenet_finegrained_rammer
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_rammer.log
popd

pushd hubert_coarse_rammer
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_rammer.log
popd

pushd hubert_coarse_int8_rammer
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_rammer.log
popd

pushd hubert_finegrained_rammer
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_rammer.log
popd


########################RAMMER-S#########################
pushd bert_coarse_rammer-s
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_rammer-s.log
popd

pushd bert_coarse_int8_rammer-s
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_rammer-s.log
popd

pushd bert_finegrained_rammer-s
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_rammer-s.log
popd


pushd mobilenet_coarse_rammer-s
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_rammer-s.log
popd

pushd mobilenet_coarse_int8_rammer-s
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_rammer-s.log
popd

pushd mobilenet_finegrained_rammer-s
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_rammer-s.log
popd

pushd hubert_coarse_rammer-s
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_rammer-s.log
popd

pushd hubert_coarse_int8_rammer-s
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_rammer-s.log
popd

pushd hubert_finegrained_rammer-s
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_rammer-s.log
popd


########################SPARTA###########################

pushd bert_coarse_sparta
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_sparta.log
popd

pushd bert_coarse_int8_sparta
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_sparta.log
popd

pushd bert_finegrained_sparta
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_sparta.log
popd

pushd mobilenet_coarse_sparta
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_sparta.log
popd

pushd mobilenet_coarse_int8_sparta
/bin/bash run.sh > ${cur_dir}/log/mobilenet_coarse_int8_sparta.log
popd

pushd mobilenet_finegrained_sparta
/bin/bash run.sh > ${cur_dir}/log/mobilenet_finegrained_sparta.log
popd

pushd hubert_coarse_sparta
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_sparta.log
popd

pushd hubert_coarse_int8_sparta
/bin/bash run.sh > ${cur_dir}/log/hubert_coarse_int8_sparta.log
popd

pushd hubert_finegrained_sparta
/bin/bash run.sh > ${cur_dir}/log/hubert_finegrained_sparta.log
popd

python analyze_log.py