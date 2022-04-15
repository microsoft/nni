cp ../../checkpoints/mobilenet/artifact_mobilenet_ori/mobilenet_ori_no_tesa.onnx .
nnfusion mobilenet_ori_no_tesa.onnx -f onnx -fkernel_tuning_steps=1000 -fantares_mode=1 -fantares_codegen_server="127.0.0.1:8881" -fir_based_fusion=true -fkernel_fusion_level=0 -frun_step 2000
pushd nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make
ln -s ../Constant
./main_test
popd
popd