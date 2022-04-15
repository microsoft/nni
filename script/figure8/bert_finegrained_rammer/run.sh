cp ../../checkpoints/bert/artifact_bert_ori/bert_ori_no_tesa.onnx .
nnfusion bert_ori_no_tesa.onnx -f onnx -flayernorm_fusion=1 -fgelu_fusion=1 -frun_step=200
pushd nnfusion_rt/cuda_codegen
mkdir build
pushd build
cmake ..
make
ln -s ../Constant
./main_test
popd
popd
