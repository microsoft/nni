from nni_client import NNIExperiment
import json
exp = NNIExperiment()
exp.connect_experiment('http://10.190.175.223:8080/')
jobs = exp.list_trial_jobs()
j = jobs[0]
for k, v in j.__dict__.items():
    print(k, v)

for k, v in j.hyperParameters[0].__dict__.items():
    print(k, v, type(v))