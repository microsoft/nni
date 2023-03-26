#!/bin/bash
set -e

if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.nni/nasbenchmark
fi

echo "Downloading NAS-Bench-201..."
if [ -f "a.pth" ]; then
    echo "a.pth found. Skip download."
else
    gdown https://drive.google.com/uc\?id\=1OOfVPpt-lA4u2HJrXbgrRd42IbfvJMyE -O a.pth
fi

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nasbench201.db ${NASBENCHMARK_DIR}/nasbench201.db-journal
mkdir -p ${NASBENCHMARK_DIR}
python3 -m nni.nas.benchmark.nasbench201.db_gen a.pth
rm -f a.pth
