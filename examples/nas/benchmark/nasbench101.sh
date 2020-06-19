set -e
mkdir -p /outputs /tmp

cd /tmp

echo "Installing dependencies..."
apt install -y wget
pip install --no-cache-dir tqdm peewee
git clone https://github.com/google-research/nasbench
cd nasbench && pip install -e . && cd ..
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
cd nni && source install.sh && cd ..

echo "Downloading NAS-Bench-201..."
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

echo "Generating database..."
rm -f /outputs/nasbench101.db /outputs/nasbench101.db-journal
NASBENCHMARK_DIR=/outputs python -m nni.nas.benchmark.nasbench101.db_gen nasbench_full.tfrecord
