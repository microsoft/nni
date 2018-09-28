## Usage

To test before installing:

    ./run.py --preinstall

To test the integrity of installation:

    ./run.py

It will print `PASS` in green eventually if everything works well.

## Details

This test case tests the communication between trials and tuner/assessor.

The naive trials receive an integer `x` as parameter, and reports `x`, `x²`, `x³`, ... , `x¹⁰` as metrics.

The naive tuner simply generates the sequence of natural numbers, and print received metrics to `tuner_result.txt`.

The naive assessor kills trials when `sum(metrics) % 11 == 1`, and print killed trials to `assessor_result.txt`.

When tuner and assessor exit without exception, they will append `DONE` to corresponding result file. Otherwise they append `ERROR`.

## Issues

* Private APIs are used to detect whether tuner and assessor have terminated successfully. 
* The output of REST server is not tested.
* Remote machine training service is not tested.
