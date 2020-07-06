set -e

echo "Downloading NAS-Bench-101..."
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nasbench101.db ${NASBENCHMARK_DIR}/nasbench101.db-journal
python -m nni.nas.benchmarks.nasbench101.db_gen nasbench_full.tfrecord
