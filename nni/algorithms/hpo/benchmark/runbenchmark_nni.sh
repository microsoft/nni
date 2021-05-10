#!/bin/bash

time=$(date "+%Y%m%d%H%M%S")
installation='automlbenchmark'
outdir="results_$time"
benchmark='nnivalid'

for tuner in 'NNI_TPE' 'NNI_RANDOM_SEARCH' 'NNI_ANNEAL' 'NNI_EVOLUTION' 'NNI_SMAC' 'NNI_GP' 'NNI_METIS' 'NNI_HYPERBAND' 'NNI_BOHB'; do
    echo "python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni"
    python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni
done

python parse_result_csv.py "$outdir/results.csv"
