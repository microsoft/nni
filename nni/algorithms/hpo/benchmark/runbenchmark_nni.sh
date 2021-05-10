#!/usr/bin/bash

outdir='results_nni'
benchmark='nnivalid'

for tuner in 'NNI_TPE' 'NNI_RANDOM_SEARCH' 'NNI_ANNEAL' 'NNI_EVOLUTION' 'NNI_SMAC' 'NNI_GP' 'NNI_METIS' 'NNI_HYPERBAND' 'NNI_BOHB'; do
    echo "python runbenchmark.py $tuner $benchmark -o $outdir -u nni"
    python runbenchmark.py $tuner $benchmark -o $outdir -u nni
done

python parse_result_csv.py "$outdir/results.csv"
