set -e
mkdir -p /outputs /tmp

cd /tmp

echo "Installing dependencies..."
apt install -y wget
pip install --no-cache-dir tqdm peewee
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
cd nni && source install.sh && cd ..

echo "Downloading NDS..."
wget https://dl.fbaipublicfiles.com/nds/data.zip -O data.zip

echo "Generating database..."
rm -f /outputs/nds.db /outputs/nds.db-journal
NASBENCHMARK_DIR=/outputs python -m nni.nas.benchmark.nds.db_gen nds_data
