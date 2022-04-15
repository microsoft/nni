cp -r ../../checkpoints/mobilenet/artifact_mobilenet_coarsegrained_no_propagation_onnx_with_tesa .
python prepare_kernel_cfg.py --in_dir artifact_mobilenet_coarsegrained_no_propagation_onnx_with_tesa --out_dir nnfusion_cfg
pushd nnfusion_cfg
nnfusion model_tesa.onnx -f onnx -fkernel_tuning_steps=1000 -fantares_mode=1 -fantares_codegen_server="127.0.0.1:8881" -fir_based_fusion=true -fkernel_fusion_level=0 -fspargen_cfg config -ftuning_blocklist="SputnikDot" -firfusion_blocklist="SputnikDot" -frun_step 2000
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
