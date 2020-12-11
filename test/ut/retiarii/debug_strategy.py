import json
import os
import logging

from nni.retiarii import Model, submit_models, wait_models



def single_model_strategy():
    with open(os.path.join(os.path.dirname(__file__), 'converted_mnist_pytorch.json')) as f:
        ir = json.load(f)
    model = Model._load(ir)
    submit_models(model)
    wait_models(model)
    print('Strategy says:', model.metric)

def multi_model_cgo():
    os.environ['CGO'] = 'true'
    with open(os.path.join(os.path.dirname(__file__), 'converted_mnist_pytorch.json')) as f:
        ir = json.load(f)
    m = Model._load(ir)
    models = [m]
    for i in range(3):
        models.append(m.fork())
    submit_models(*models)
    wait_models(*models)
    
    print('Strategy says:', [_.metric for _ in models])

if __name__ == '__main__':
    single_model_strategy()
