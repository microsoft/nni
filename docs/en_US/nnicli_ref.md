# NNI Client

NNI client is a python API of `nnictl`, which implements the most commonly used commands. Users can use this API to control their experiments, collect experiment results and conduct advanced analyses based on experiment results in python code directly instead of using command line. Here is an example:

```
from nni.experiment import Experiment

# create an experiment instance
exp = Experiment() 

# start an experiment, then connect the instance to this experiment
# you can also use `resume_experiment`, `view_experiment` or `connect_experiment`
# only one of them should be called in one instance
exp.start_experiment('nni/examples/trials/mnist-pytorch/config.yml', port=9090)

# update the experiment's concurrency
exp.update_concurrency(3)

# get some information about the experiment
print(exp.get_experiment_status())
print(exp.get_job_statistics())
print(exp.list_trial_jobs())

# stop the experiment, then disconnect the instance from the experiment.
exp.stop_experiment()
```

## References

```eval_rst
..  autoclass:: nni.experiment.Experiment
    :members:
..  autoclass:: nni.experiment.TrialJob
    :members:
..  autoclass:: nni.experiment.TrialHyperParameters
    :members:
..  autoclass:: nni.experiment.TrialMetricData
    :members:
..  autoclass:: nni.experiment.TrialResult
    :members:
```
