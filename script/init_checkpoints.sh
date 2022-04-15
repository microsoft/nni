source /root/anaconda/etc/profile.d/conda.sh
conda activate artifact

pushd checkpoints/bert
bash prepare_data.sh
pip install transformers==3.5.0
python bert_propagate_finegrained.py > finegrained.log
python bert_propagate_coarsegrained.py > coarse.log
python bert_sota_coarse_onnx.py
python bert_sota_finegrained_onnx.py
python bert_original_onnx.py
python bert_mix_onnx.py
popd

pushd checkpoints/mobilenet
# prepare the data
echo "Prepare the Imagenet-Dot Dataset"
bash prepare_data.sh
python mobilenet_propagate_coarsegrained.py > coarse.log
python mobilenet_propagate_finegrained.py > finegrained.log
python mobilenet_sota_finegrained_onnx.py
python mobilenet_sota_coarsegrained_onnx.py
python mobilenet_ori_onnx.py
popd

pip install transformers==4.12.3
pushd checkpoints/hubert
bash run_coarse.sh > coarse.log
bash run_finegrained.sh > finegrained.log
bash run_ori_onnx.sh
bash run_finegrained_sota.sh
bash run_coarse_sota.sh
popd