set -e

echo "Downloading NDS..."
wget https://dl.fbaipublicfiles.com/nds/data.zip -O data.zip
unzip data.zip

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nds.db ${NASBENCHMARK_DIR}/nds.db-journal
python -m nni.nas.benchmarks.nds.db_gen nds_data
