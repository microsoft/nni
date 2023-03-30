#!/bin/bash
set -e

if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.nni/nasbenchmark
fi

echo "Downloading NAS-Bench-101..."
if [ -f "nasbench_full.tfrecord" ]; then
    echo "nasbench_full.tfrecord found. Skip download."
else
    wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
fi

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nasbench101.db ${NASBENCHMARK_DIR}/nasbench101.db-journal
mkdir -p ${NASBENCHMARK_DIR}
python3 -m nni.nas.benchmark.nasbench101.db_gen nasbench_full.tfrecord
rm -f nasbench_full.tfrecord
