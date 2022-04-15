cp ../../checkpoints/hubert/artifact_hubert_ori/hubert_ori_no_tesa.onnx . 
nnfusion hubert_ori_no_tesa.onnx -f onnx -frun_step 300
pushd nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make
ln -s ../Constant
./main_test
popd
popd
