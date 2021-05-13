#!/bin/bash

time=$(date "+%Y%m%d%H%M%S")
installation='automlbenchmark'
outdir="results_$time"
benchmark='nnismall'

if [ "$#" -eq 0 ]; then
    tuner_array=('TPE' 'Random' 'Anneal' 'Evolution' 'GPTuner' 'MetisTuner' 'Hyperband')
else
    tuner_array=( "$@" )
fi

for tuner in ${tuner_array[*]}; do
    echo "python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni"
    python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni
done

python parse_result_csv.py "$outdir/results.csv"
