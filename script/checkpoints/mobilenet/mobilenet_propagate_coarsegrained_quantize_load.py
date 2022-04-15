
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet_utils import *
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

def train(model, device, train_loader, optimizer):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    loss_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            loss_list.append(loss)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = np.array(loss_list).mean()

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))
    return 100 * correct / len(test_loader.dataset)

def get_mobile_coarse():
    import torch
    from nni.compression.pytorch.speedup import ModelSpeedup
    model = create_model('mobilenet_v1')
    dummy_input = torch.rand(1,3,224,224)
    new_model = align_speedup(model, dummy_input, 'checkpoints/coarsegrained/mobilenet_0.6_align_run1/mask_temp.pth')
    state = torch.load('checkpoints/coarsegrained/mobilenet_0.6_align_run1/finetune_weights.pth')
    new_model.load_state_dict(state)
    return new_model

m = get_mobile_coarse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TrainDataset('./data/stanford-dogs/Processed/train')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(run_eval(m.cuda(), test_dataloader, device))
print('Propagation done')

configure_list = [{
    'quant_types': ['weight'],
    'quant_bits': {
        'weight': 8
    }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
    'op_types':['Conv2d']
},{
    'quant_types': ['output'],
    'quant_bits': {
        'output': 8
    }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
    'op_types':['ReLU']
}]

#set_quant_scheme_dtype('weight', 'per_tensor_symmetric', 'int')
#set_quant_scheme_dtype('output', 'per_tensor_symmetric', 'int')
#set_quant_scheme_dtype('input', 'per_tensor_symmetric', 'int')

model = m.cuda()
dummy_input = torch.rand(1,3,224,224).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.5)

model_path = "mobilenet_qat.pth"
calibration_path = "mobilenet_qat_calibration.pth"
onnx_path = "mobilenet_qat.onnx"
input_shape = (1,3,224,224)

import pdb; pdb.set_trace()

model.load_state_dict(torch.load(model_path))

quantizer = QAT_Quantizer(model, configure_list, optimizer)
quantizer.compress()

mobilenet_ptq_calibration = torch.load(calibration_path)
quantizer.load_calibration_config(mobilenet_ptq_calibration)

print("load ptq calibration config successfully")
test(model, device, test_dataloader)