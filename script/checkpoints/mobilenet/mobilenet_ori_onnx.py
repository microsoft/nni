from torch import device
from mobilenet_utils import *
import torch
import os

device = torch.device('cuda')

model = create_model('mobilenet_v1').to(device)
data = torch.rand(32, 3, 224, 224).to(device)
prefix = 'artifact_mobilenet_ori'
os.makedirs(prefix, exist_ok=True)
torch.onnx.export(model, data, os.path.join(prefix, 'mobilenet_ori_no_tesa.onnx'))