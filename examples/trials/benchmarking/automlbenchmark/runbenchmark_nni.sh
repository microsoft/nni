#!/bin/bash

time=$(date "+%Y%m%d%H%M%S")
installation='automlbenchmark'
outdir="results_$time"
benchmark='nnivalid'      # 'nnismall' 'nnismall-regression' 'nnismall-binary' 'nnismall-multiclass' 
serialize=true           # if false, run all experiments together in background

mkdir $outdir $outdir/scorelogs $outdir/reports 

if [ "$#" -eq 0 ]; then
    tuner_array=('TPE' 'Random' 'Anneal' 'Evolution' 'GPTuner' 'MetisTuner' 'Hyperband')
else
    tuner_array=( "$@" )
fi

if [ "$serialize" = true ]; then
    # run tuners serially 
    for tuner in ${tuner_array[*]}; do
	echo "python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni"
	python $installation/runbenchmark.py $tuner $benchmark -o $outdir -u nni
    done

    # parse final results
    echo "python parse_result_csv.py $outdir/results.csv"
    python parse_result_csv.py "$outdir/results.csv"

else
    # run all the tuners in background
    for tuner in ${tuner_array[*]}; do
	mkdir "$outdir/$tuner" "$outdir/$tuner/scorelogs"
	echo "python $installation/runbenchmark.py $tuner $benchmark -o $outdir/$tuner -u nni &"
	python $installation/runbenchmark.py $tuner $benchmark -o $outdir/$tuner -u nni &
    done
    
    wait

    # aggregate results
    touch "$outdir/results.csv"
    let i=0
    for tuner in ${tuner_array[*]}; do
	cp "$outdir/$tuner/scorelogs"/* $outdir/scorelogs
	if [ $i -eq 0 ]; then
	    cp "$outdir/$tuner/results.csv" "$outdir/results.csv"
	else
	    let nlines=`cat "$outdir/$tuner/results.csv" | wc -l`
	    ((nlines=nlines-1))
	    tail -n $nlines "$outdir/$tuner/results.csv" >> "$outdir/results.csv" 
	fi
	((i=i+1))
    done

    # parse final results
    echo "python parse_result_csv.py $outdir/results.csv"
    python parse_result_csv.py "$outdir/results.csv"
fi
