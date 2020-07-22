from nni_client import *
import nnicli as nc
nc.set_endpoint('http://localhost:8080')
print(nc.get_trial_job('nDen7'))
# print(nc.get_job_metrics('aaa-8'))