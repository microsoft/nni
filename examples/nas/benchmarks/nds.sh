#!/bin/bash
set -e

if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.nni/nasbenchmark
fi

echo "Downloading NDS..."
if [ -f "data.zip" ]; then
    echo "data.zip found. Skip download."
else
    wget https://dl.fbaipublicfiles.com/nds/data.zip -O data.zip
fi
unzip data.zip

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nds.db ${NASBENCHMARK_DIR}/nds.db-journal
mkdir -p ${NASBENCHMARK_DIR}
python3 -m nni.nas.benchmark.nds.db_gen nds_data
rm -rf data.zip nds_data
