

from bohb_advisor import BOHB


search_space = {
"lr":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
"optimizer":{"_type":"choice", "_value":["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"]},
"model":{"_type":"choice", "_value":["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "senet18"]}
}


bohb = BOHB()
bohb.handle_initialize(search_space)
bohb.handle_request_trial_jobs(5)