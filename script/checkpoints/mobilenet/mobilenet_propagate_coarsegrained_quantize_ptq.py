
import torch
import torch.nn.functional as F
from mobilenet_utils import *
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.algorithms.compression.pytorch.quantization import ObserverQuantizer
from nni.compression.pytorch.quantization.settings import set_quant_scheme_dtype

def calibration(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            model(data)

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)
    print('Loss: {}  accuracy: {}%)\n'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))
    return acc

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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.5)

# quantizer = QAT_Quantizer(model, configure_list, optimizer, dummy_input=dummy_input)
quantizer = ObserverQuantizer(model.eval(), configure_list, optimizer)
calibration(model, device, test_dataloader)
quantizer.compress()

acc = test(model, device, test_dataloader)

print(f"Accuracy:{acc}")