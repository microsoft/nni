from torch import device
from mobilenet_utils import *
import torch
from sparta.common.utils import measure_time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=200)
args = parser.parse_args()

device = torch.device('cuda')

model = create_model('mobilenet_v1').to(device)
data = torch.rand(32, 3, 224, 224).to(device)
jit_model = torch.jit.trace(model, data)
del model
time_mean, time_std = measure_time(jit_model, [data], args.iterations)
print('RunningTime = {}'.format(time_mean))