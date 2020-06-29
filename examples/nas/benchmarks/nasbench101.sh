set -e
mkdir -p /outputs /tmp

echo "Installing dependencies..."
apt update && apt install -y wget git
pip install --no-cache-dir tqdm peewee

echo "Installing NNI..."
cd /nni && echo "y" | source install.sh

cd /tmp

echo "Installing NASBench..."
git clone https://github.com/google-research/nasbench
cd nasbench && pip install -e . && cd ..

echo "Downloading NAS-Bench-101..."
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

echo "Generating database..."
rm -f /outputs/nasbench101.db /outputs/nasbench101.db-journal
NASBENCHMARK_DIR=/outputs python -m nni.nas.benchmarks.nasbench101.db_gen nasbench_full.tfrecord
