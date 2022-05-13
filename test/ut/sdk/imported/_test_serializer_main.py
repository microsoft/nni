import sys
import torch.nn as nn

# sys.argv[1] == 0 -> dump
# sys.argv[1] == 1 -> load

import nni
from nni.retiarii import model_wrapper

@model_wrapper
class Net(nn.Module):
    something = 1

if sys.argv[1] == '0':
    # This could be extraordinary large on MacOS
    nni.dump(Net, fp=open('serialize_result.txt', 'w'), pickle_size_limit=16384)
else:
    obj = nni.load(fp=open('serialize_result.txt'))
    assert obj().something == 1
