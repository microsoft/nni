#!/bin/bash
set -e

if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.nni/nasbenchmark
fi


mkdir -p nlp_data
cd nlp_data
echo "Downloading NLP[1/2] w_data.zip..."
if [ -f "w_data.zip" ]; then
    echo "w_data.zip found. Skip download."
else
    wget -O w_data.zip https://github.com/fmsnew/nas-bench-nlp-release/blob/master/train_logs_wikitext-2/logs.zip?raw=true
fi
unzip w_data.zip
echo "Downloading NLP[2/2] ptb_data.zip..."
if [ -f "ptb_data.zip" ]; then
    echo "ptb_data.zip found. Skip download."
else
    wget -O ptb_data.zip https://github.com/fmsnew/nas-bench-nlp-release/blob/master/train_logs_multi_runs/logs.zip?raw=true
fi
unzip ptb_data.zip
cd ..

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nlp.db ${NASBENCHMARK_DIR}/nlp.db-journal
mkdir -p ${NASBENCHMARK_DIR}
python3 -m nni.nas.benchmarks.nlp.db_gen nlp_data
rm -rf nlp_data
