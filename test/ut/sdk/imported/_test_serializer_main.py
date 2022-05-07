import sys
import torch.nn as nn

# sys.argv[1] == 0 -> dump
# sys.argv[1] == 1 -> load

import nni
from nni.retiarii import model_wrapper

@model_wrapper
class Net(nn.Module):
    something = 1

import cloudpickle

if sys.argv[1] == '0':
    cloudpickle.dump(Net, open('serialize_result.txt', 'wb'))
    # nni.dump(Net, fp=open('serialize_result.txt', 'w'))
else:
    obj = cloudpickle.load(open('serialize_result.txt', 'rb'))
    # obj = nni.load(fp=open('serialize_result.txt'))
    assert obj().something == 1
