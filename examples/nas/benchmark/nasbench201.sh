set -e
mkdir -p /outputs /tmp

echo "Installing NNI..."
cd /nni && echo "y" | source install.sh

cd /tmp

echo "Installing dependencies..."
pip install --no-cache-dir gdown tqdm peewee

echo "Downloading NAS-Bench-201..."
gdown https://drive.google.com/uc\?id\=1OOfVPpt-lA4u2HJrXbgrRd42IbfvJMyE -O a.pth

echo "Generating database..."
rm -f /outputs/nasbench201.db /outputs/nasbench201.db-journal
NASBENCHMARK_DIR=/outputs python -m nni.nas.benchmark.nasbench201.db_gen a.pth
