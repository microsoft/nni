# NNI Client

NNI client is a python API of `nnictl`, which implements the most common used commands. User can use this API to control their experiments in python code instead of command line. Here is an example:

```
import nnicli as nc

nc.start_experiment('nni/examples/trials/mnist-pytorch/config.yml', port=9090) # start an experiment

nc.set_endpoint('http://localhost:9090') # set the experiment's endpoint, i.e., the url of Web UI

print(nc.version()) # check the version of nni
print(nc.get_experiment_status()) # get the experiment's status

print(nc.get_job_statistics()) # get the trial job information
print(nc.list_trial_jobs()) # get information for all trial jobs

nc.stop_nni(port=9090) # stop the experiment
```

## References

```eval_rst
.. autofunction:: nnicli.start_experiment
.. autofunction:: nnicli.set_endpoint
.. autofunction:: nnicli.resume_experiment
.. autofunction:: nnicli.view_experiment
.. autofunction:: nnicli.update_searchspace
.. autofunction:: nnicli.update_concurrency
.. autofunction:: nnicli.update_duration
.. autofunction:: nnicli.update_trailnum
.. autofunction:: nnicli.stop_experiment
.. autofunction:: nnicli.version
.. autofunction:: nnicli.get_experiment_status
.. autofunction:: nnicli.get_experiment_profile
.. autofunction:: nnicli.get_trial_job
.. autofunction:: nnicli.list_trial_jobs
.. autofunction:: nnicli.get_job_statistics
.. autofunction:: nnicli.get_job_metrics
.. autofunction:: nnicli.export_data
```