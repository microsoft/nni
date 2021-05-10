# Benchmark for Tuners

This folder contains a benchmark tool that enables comparisons between the performances of tuners provided by NNI (and users' custom tuners) on different tasks. The implementation of this tool is based on the automlbenchmark repository (https://github.com/openml/automlbenchmark), which provides services of running different *frameworks* against different *benchmarks* consisting of multiple *tasks*. 

### Terminology

* #task: a task can be thought of as (dataset, evaluator). It gives out a dataset containing (train, valid, test), and based on the received predictions, the evaluator evaluates a given metric (e.g., mse for regression, f1 for classfication). 
* benchmark: a benchmark is a set of tasks, along with other external constraints such as time and resource. 
* #framework: given a task, a framework conceives answers to the proposed regression or classification problem and produces predictions. Note that the automlbenchmark framework does not pose any restrictions on the hypothesis space of a framework. In our implementation in this folder, each framework is a tuple (tuner, architecture), where architecture provides the hypothesis space (and search space for tuner), and tuner determines the strategy of hyperparameter optimization. 
* #tuner: a tuner or advisor defined in the hpo folder, or a custom tuner provided by the user. 
* #architecture: an architecture is a specific method for solving the tasks, along with a set of hyperparameters to optimize (i.e., the search space). In our implementation, the architecture calls tuner multiple times to obtain possible hyperparameter configurations, and produces the final prediction for a task. See `./nni/extensions/NNI/architectures` for examples.

### Setup
```bash
./setup.sh
```

### Run predefined benchmarks on existing tuners
```bash
./runbenchmark_nni.sh
```
This script runs the benchmark 'nnivalid', which consists of a regression task, a binary classification task, and a multi-class classification task. After the script finishes, you can find a summary of the results in the "results.csv.parsed" file in the result folder associated with current time. To run on other predefined benchmarks, change the `benchmark` variable in `runbenchmark_nni.sh`. Some benchmarks are defined in `./nni/benchmarks`, and others are defined in `./automlbenchmark/resources/benchmarks/`.

### Run customized benchmarks on existing tuners
To run customized benchmarks, add a benchmark_name.yaml file in the folder `./nni/benchmarks`, and change the `benchmark` variable in `runbenchmark_nni.sh`. See `./automlbenchmark/resources/benchmarks/` for some examples of defining a custom benchmark.

### Run benchmarks on custom tuners
To use custom tuners, first make sure that the tuner inherits from `nni.tuner.Tuner` and correctly implements the required APIs. Next, perform the following steps:
1. Copy the implementation into the folder `./nni/extensionsions/NNI`.
1. In `tuners.py` (under the same folder), import the custom tuner, and change the `get_tuner` function to include the new option.
1. In `./nni/frameworks.yaml`, add a new framework extending the base framework NNI. Make sure that the parameter `tuner_type` corresponds to the name in step 2.
1. (Optional) Include the new framework in the last step into `runbenchmark_nni.sh`. 
