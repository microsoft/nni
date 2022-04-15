# measure the jit time
cur_dir=`pwd`
source ~/.bashrc
echo "Curret directory ${cur_dir}"
mkdir ${cur_dir}/log

source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

pushd baseline
bash run.sh > ${cur_dir}/log/bert_baseline.log
popd


pushd bert_coarse
pushd sparse_kernel
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_sparse_kernel.log
popd
popd

pushd bert_coarse
pushd propagation
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_propagation.log
popd
popd

pushd bert_coarse
pushd transformation
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_transformation.log
popd
popd

pushd bert_coarse
pushd specialization
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_specialization.log
popd
popd

pushd bert_coarse_int8
pushd sparse_kernel
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_sparse_kernel.log
popd
popd

pushd bert_coarse_int8
pushd propagation
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_propagation.log
popd
popd

pushd bert_coarse_int8
pushd transformation
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_transformation.log
popd
popd

pushd bert_coarse_int8
pushd specialization
/bin/bash run.sh > ${cur_dir}/log/bert_coarse_int8_specialization.log
popd
popd

pushd bert_mixed
pushd sparse_kernel
/bin/bash run.sh > ${cur_dir}/log/bert_mixed_sparse_kernel.log
popd
popd

pushd bert_mixed
pushd propagation
/bin/bash run.sh > ${cur_dir}/log/bert_mixed_propagation.log
popd
popd

pushd bert_mixed
pushd transformation
/bin/bash run.sh > ${cur_dir}/log/bert_mixed_transformation.log
popd
popd

pushd bert_mixed
pushd specialization
/bin/bash run.sh > ${cur_dir}/log/bert_mixed_specialization.log
popd
popd

pushd bert_finegrained
pushd sparse_kernel
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_sparse_kernel.log
popd
popd

pushd bert_finegrained
pushd propagation
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_propagation.log
popd
popd

pushd bert_finegrained
pushd transformation
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_transformation.log
popd
popd

pushd bert_finegrained
pushd specialization
/bin/bash run.sh > ${cur_dir}/log/bert_finegrained_specialization.log
popd
popd

python analyze_log.py