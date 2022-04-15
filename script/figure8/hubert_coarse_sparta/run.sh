cp -r ../../checkpoints/hubert/artifact_hubert_coarse_onnx_with_tesa .
mkdir nnfusion_cfg
cp artifact_hubert_coarse_onnx_with_tesa/model_tesa.onnx nnfusion_cfg
cp artifact_hubert_coarse_onnx_with_tesa/model_no_tesa.onnx nnfusion_cfg
pushd nnfusion_cfg
# nnfusion model_tesa.onnx -f onnx
tuning_step=1000
nnfusion model_no_tesa.onnx -f onnx -fkernel_tuning_steps=$tuning_step -fantares_mode=1 -fantares_codegen_server="127.0.0.1:8881" -fir_based_fusion=true -fkernel_fusion_level=0 -fblockfusion_level=1 -ftuning_blocklist="Dot,QuantizeDot,InstanceNorm,Softmax,Convolution" -firfusion_blocklist="Dot,QuantizeDot,InstanceNorm,Softmax,Convolution" -frun_step=500

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

