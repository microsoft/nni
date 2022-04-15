cp -r ../../checkpoints/bert/artifact_bert_coarse_onnx_with_tesa .
python bert_codegen.py
rm ~/.cache/nnfusion/kernel_cache.db
python prepare_kernel_cfg.py --in_dir ../../checkpoints/bert/artifact_bert_coarse_onnx_with_tesa --out_dir ./nnfusion_cfg_bert_coarse
cp artifact_bert_coarse_onnx_with_tesa/model_tesa.onnx nnfusion_cfg_bert_coarse
pushd ./nnfusion_cfg_bert_coarse
nnfusion model_tesa.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1 -fspargen_cfg config -frun_step=500
#nnfusion model.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1 -fblockfusion_level=0 -fcodegen_debug=true -fspargen_cfg config
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

