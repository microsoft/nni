mkdir -p ~/.cache/nnfusion/
rm ~/.cache/nnfusion/kernel_cache.db
cp -r ../../checkpoints/hubert/artifact_hubert_finegrained_onnx_with_tesa/ .
python prepare_kernel_cfg.py
cp kernel_cache.db ~/.cache/nnfusion/kernel_cache.db
mkdir nnfusion_cfg
pushd nnfusion_cfg

nnfusion ../artifact_hubert_finegrained_onnx_with_tesa/model_tesa.onnx -f onnx -fsparta=true -fkernels_as_files=true -fkernels_files_number=40 -fgelu_fusion=true -fsparse_dot_transpose=true -frun_step 300
pushd nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make -j
ln -s ../Constant
./main_test
popd
popd
popd
