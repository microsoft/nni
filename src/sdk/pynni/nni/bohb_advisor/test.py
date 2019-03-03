from bohb_advisor import BOHB

search_space = {
"lr":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
"optimizer":{"_type":"choice", "_value":["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"]},
"model":{"_type":"choice", "_value":["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "senet18"]}
}


bohb = BOHB()
bohb.handle_initialize(search_space)

bohb.handle_request_trial_jobs(5)

result1 = dict()
result1['parameter_id'] = '1_0_0'
result1['value'] = 0.74
result1['trial_job_id'] = '666'
result1['type'] = 'FINAL'
result1['sequence'] = 20
bohb.handle_report_metric_data(result1)
data1 = dict()
data1['trial_job_id'] = '666'
data1['event'] = 'SUCCESS'
data1['hyper_params'] = "{'STEPS': 1, 'model': 'googlenet', 'optimizer': 'Adam', 'lr': 0.1}"
bohb.handle_trial_end(data1)

result2 = dict()
result2['parameter_id'] = '1_0_1'
result2['value'] = 0.59
result2['trial_job_id'] = '233'
result2['type'] = 'FINAL'
result2['sequence'] = 20
bohb.handle_report_metric_data(result2)
data2 = dict()
data2['trial_job_id'] = '233'
data2['event'] = 'SUCCESS'
data2['hyper_params'] = "{'STEPS': 1, 'model': 'densenet121', 'optimizer': 'Adadelta', 'lr': 0.01}"
bohb.handle_trial_end(data2)

result3 = dict()
result3['parameter_id'] = '1_0_2'
result3['value'] = 0.13
result3['trial_job_id'] = '999'
result3['type'] = 'FINAL'
result3['sequence'] = 20
bohb.handle_report_metric_data(result3)
data3 = dict()
data3['trial_job_id'] = '999'
data3['event'] = 'SUCCESS'
data3['hyper_params'] = "{'STEPS': 1, 'model': 'senet18', 'optimizer': 'Adam', 'lr': 0.01}"
bohb.handle_trial_end(data3)

bohb.handle_request_trial_jobs(5)