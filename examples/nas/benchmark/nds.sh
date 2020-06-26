set -e
mkdir -p /outputs /tmp

echo "Installing dependencies..."
apt update && apt install -y wget
pip install --no-cache-dir tqdm peewee

echo "Installing NNI..."
cd /nni && echo "y" | source install.sh

cd /tmp

echo "Downloading NDS..."
wget https://dl.fbaipublicfiles.com/nds/data.zip -O data.zip

echo "Generating database..."
rm -f /outputs/nds.db /outputs/nds.db-journal
NASBENCHMARK_DIR=/outputs python -m nni.nas.benchmark.nds.db_gen nds_data
