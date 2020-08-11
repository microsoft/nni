from nni_client import NNIExperiment
import json
import time
exp = NNIExperiment()
exp.start_experiment('/home/v-junwsu/mnist-pytorch/config.yml')
time.sleep(10)
jobs = exp.list_trial_jobs()
j = jobs[0]
for k, v in j.__dict__.items():
    print(k, v)

for k, v in j.hyperParameters[0].__dict__.items():
    print(k, v, type(v))