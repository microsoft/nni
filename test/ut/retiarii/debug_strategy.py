import json
import os

from nni.retiarii import Model, submit_models, wait_models


def single_model_startegy():
    with open(os.path.join(os.path.dirname(__file__), 'mnist_pytorch.json')) as f:
        ir = json.load(f)
    model = Model._load(ir)
    submit_models(model)
    wait_models(model)
    print('Strategy says:', model.metric)


if __name__ == '__main__':
    single_model_startegy()
