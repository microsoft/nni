set -e
mkdir -p /outputs /tmp

cd /tmp

echo "Installing dependencies..."
pip install --no-cache-dir gdown tqdm peewee
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
cd nni && source install.sh && cd ..

echo "Downloading NAS-Bench-201..."
gdown https://drive.google.com/uc\?id\=1OOfVPpt-lA4u2HJrXbgrRd42IbfvJMyE -O a.pth

echo "Generating database..."
NASBENCHMARK_DIR=/outputs python -m nasbench201.db_gen a.pth
