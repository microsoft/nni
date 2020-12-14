#!/bin/bash
set -e

if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.nni/nasbenchmark
fi


mkdir -p nlp_data
cd nlp_data
echo "Downloading NLP[1/3] wikitext2_data.zip..."
if [ -f "wikitext2_data.zip" ]; then
    echo "wikitext2_data.zip found. Skip download."
else
    wget -O wikitext2_data.zip https://github.com/fmsnew/nas-bench-nlp-release/blob/master/train_logs_wikitext-2/logs.zip?raw=true
fi
echo "Downloading NLP[2/3] ptb_single_run_data.zip..."
if [ -f "ptb_single_run_data.zip" ]; then
    echo "ptb_single_run_data.zip found. Skip download."
else
    wget -O ptb_single_run_data.zip https://github.com/fmsnew/nas-bench-nlp-release/blob/master/train_logs_single_run/logs.zip?raw=true
fi
echo "Downloading NLP[3/3] ptb_multi_runs_data.zip..."
if [ -f "ptb_multi_runs_data.zip" ]; then
    echo "ptb_multi_runs_data.zip found. Skip download."
else
    wget -O ptb_multi_runs_data.zip https://github.com/fmsnew/nas-bench-nlp-release/blob/master/train_logs_multi_runs/logs.zip?raw=true
fi
echo "### there exits duplicate log_files in ptb_single_run_data.zip and ptb_multi_run_data.zip, you can ignore all or replace all ###"
unzip -q wikitext2_data.zip
unzip -q ptb_single_run_data.zip
unzip -q ptb_multi_runs_data.zip
cd ..

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nlp.db ${NASBENCHMARK_DIR}/nlp.db-journal
mkdir -p ${NASBENCHMARK_DIR}
python3 -m nni.nas.benchmarks.nlp.db_gen nlp_data
rm -rf nlp_data
