#!/bin/bash

time=$(date "+%Y%m%d%H%M%S")
installation='automlbenchmark'
outdir="results_$time"
benchmark='nnivalid'

if [ "$#" -eq 0 ]; then
    tuner_array=('NNI_TPE' 'NNI_RANDOM_SEARCH' 'NNI_ANNEAL' 'NNI_EVOLUTION' 'NNI_SMAC' 'NNI_GP' 'NNI_METIS' 'NNI_HYPERBAND' 'NNI_BOHB')
else
    tuner_array=( "$@" )
fi

for tuner in ${tuner_array[*]}; do
    echo "python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni"
    python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni
done

python parse_result_csv.py "$outdir/results.csv"
