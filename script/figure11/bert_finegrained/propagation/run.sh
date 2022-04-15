cp -r ../../../checkpoints/bert/artifact_bert_finegrained_onnx_with_tesa .
python prepare_kernel_cfg.py --in_dir artifact_bert_finegrained_onnx_with_tesa --out_dir nnfusion_cfg
cp CMakeLists.txt nnfusion_cfg
pushd nnfusion_cfg

nnfusion model_tesa.onnx -f onnx -fspargen_cfg config -fgelu_fusion=1 -fsparse_dot_transpose=true -frun_step 300
cp CMakeLists.txt nnfusion_rt/cuda_codegen
pushd nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make
ln -s ../Constant
./main_test
popd
popd
popd
