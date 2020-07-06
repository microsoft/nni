set -e

echo "Downloading NAS-Bench-201..."
gdown https://drive.google.com/uc\?id\=1OOfVPpt-lA4u2HJrXbgrRd42IbfvJMyE -O a.pth

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nasbench201.db ${NASBENCHMARK_DIR}/nasbench201.db-journal
python -m nni.nas.benchmarks.nasbench201.db_gen a.pth
