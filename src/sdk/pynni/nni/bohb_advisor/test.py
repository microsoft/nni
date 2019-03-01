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
result1['parameter_id'] = '1_0_1'
result1['value'] = 0.74
result1['trial_job_id'] = '666'
result1['type'] = 'FINAL'
result1['sequence'] = 20

bohb.handle_report_metric_data(result1)