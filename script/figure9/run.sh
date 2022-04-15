
source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact


pip install transformers==3.5.0
python measure_memory.py --model bert
python measure_memory.py --model mobilenet
pip install transformers==4.12.3
python measure_memory.py --model hubert

# draw figure9
python draw.py