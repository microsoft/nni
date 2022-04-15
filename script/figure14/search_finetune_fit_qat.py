from typing import Dict
from numpy.core.fromnumeric import choose
from pygments.lexer import default
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import json
import math
import os
# from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer, LsqQuantizer, ObserverQuantizer
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer, ObserverQuantizer
import argparse
from torch.quantization import RecordingObserver as _RecordingObserver

class RecordingObserver(_RecordingObserver):
    """
    A extended version of PyTorch's RecordingObserver, used to record gpu tensor
    """

    def forward(self, x):
        val = x.cpu()
        super().forward(val)
        return x

class Model_conv(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.conv = module

    def forward(self, x):
        return self.conv(x)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()

feature_size = model.classifier[1].weight.data.size()[1]
replace_classifier = torch.nn.Linear(feature_size, 120)
model.classifier[1] = replace_classifier

from utils import TrainDataset, DataLoader, EvalDataset

def generate_bit_setting():
    bit_setting = [[8, -1]]
    for i in range(52):
        bit_setting.append([8, 8])
    return bit_setting

mix_bit_setting = [[8, -1], [7, 6], [7, 4], [7, 3], [6, 4], [6, 4], [6, 5], [5, 4], [6, 4], [6, 5], [6, 4], [6, 5], [6, 5], [4, 3], [6, 5], [6, 6], [5, 3], [6, 5], [6, 6], [6, 4], [6, 5], [6, 6], [6, 5], [6, 6], [6, 6], [6, 5], [6, 6], [6, 6], [6, 5], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 5], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [8, 8]]
mix_bit_setting_cand1 = [[8, -1], [7, 6], [7, 5], [7, 4], [7, 4], [6, 4], [6, 4], [5, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 5], [5, 3], [6, 5], [6, 5], [5, 3], [6, 5], [6, 5], [6, 5], [6, 5], [6, 6], [6, 5], [6, 6], [6, 6], [6, 5], [6, 6], [6, 6], [6, 5], [6, 6], [6, 6], [6, 5], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [8, 8]]
mix_bit_setting_cand2 = [[8, -1], [5, 4], [6, 4], [6, 4], [6, 4], [6, 4], [7, 5], [4, 3], [7, 4], [7, 5], [7, 4], [7, 5], [7, 7], [5, 4], [7, 5], [7, 7], [6, 4], [7, 5], [7, 7], [7, 5], [7, 6], [7, 7], [7, 6], [7, 7], [7, 7], [7, 5], [6, 6], [6, 6], [6, 5], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [8, 8]]
mix_bit_setting_cand3 = [[8, -1], [5, 4], [5, 4], [5, 4], [7, 5], [7, 4], [7, 5], [4, 3], [7, 4], [7, 5], [7, 3], [7, 5], [7, 7], [5, 4], [7, 7], [7, 7], [5, 3], [7, 7], [7, 7], [7, 4], [7, 7], [7, 7], [7, 6], [7, 7], [7, 7], [7, 6], [7, 7], [7, 7], [7, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [8, 8]]
mix_bit_setting_cand4 = generate_bit_setting()
mix_bit_setting_cand5 = [[8, 8], [5, 6], [5, 5], [5, 6], [7, 5], [7, 5], [7, 5], [4, 6], [7, 5], [7, 5], [7, 6], [7, 5], [7, 7], [5, 6], [7, 7], [7, 7], [5, 5], [7, 7], [7, 7], [7, 4], [7, 7], [7, 7], [7, 6], [7, 7], [7, 7], [7, 6], [7, 7], [7, 7], [7, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [8, 8]]
mininal_weight_bits = [8, 4, 4, 3, 4, 3, 4, 4, 4, 4, 3, 3, ]

batch_size = 32

train_dataset = TrainDataset('./data/stanford-dogs/Processed/train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataset_for_pruner = EvalDataset('./data/stanford-dogs/Processed/train')
train_dataloader_for_pruner = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = EvalDataset('./data/stanford-dogs/Processed/valid')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def _setattr(model, name, module):
    """
    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization
    name : str
        name of pytorch module
    module : torch.nn.Module
        Layer module of pytorch model
    """
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def generate_configure_list(model, bit_setting, quant_start_step=0):
    idx = 0
    configure_list = []
    support_op_types = [torch.nn.Conv2d, torch.nn.Linear]
    for name, module in model.named_modules():
        if type(module) in support_op_types:
            config = {}
            config['op_names'] = [name]
            config['quant_types'] = ['weight', 'output'] if bit_setting[idx][1] != -1 else ['weight']
            config['quant_bits'] = {'weight': bit_setting[idx][0], 'output': bit_setting[idx][1]} if bit_setting[idx][1] != -1 \
                else {'weight': bit_setting[idx][0]}
            config['quant_start_step'] = quant_start_step
            configure_list.append(config)
            idx += 1
    return configure_list

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
            print(f'current steps: {model.steps}')

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

def finetune_before_quantize(model, device, epoch, train_dataloader, test_dataloader):
    # load state dict from pretrained model path
    PATH = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_original_model/pretrained_mobilenet_v2_best_checkpoint.pt"
    model.load_state_dict(torch.load(PATH))

    best_acc = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    for i in range(epoch):
        print(f"Begin to finetune epoch {i}")
        train(model, device, train_dataloader, optimizer)
        acc = test(model, device, test_dataloader)
        if acc > best_acc:
            best_acc = acc
            save_path = "finetune_original_model/mobilenetv2_finetune.pth"
            torch.save(model.state_dict(), save_path)
            print(f"finetuned model has been saved to {save_path}")

def finetune_post_training_quantize(model, device, train_dataloader, test_dataloader):
    original_checkpoint = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_original_model/pretrained_mobilenet_v2_best_checkpoint.pt"
    model.load_state_dict(torch.load(original_checkpoint))
    configure_config = generate_configure_list(model, mix_bit_setting_cand4, quant_start_step=1000)
    print(configure_config)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)
    model.to(device)
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    quantizer = ObserverQuantizer(model.eval(), configure_config, optimizer)
    # quantizer = QAT_Quantizer(model, configure_config, optimizer, dummy_input=dummy_input)
    # quantizer = LsqQuantizer(model, configure_config, optimizer, dummy_input=dummy_input)

    def calibration(model, device, test_loader):
        model.eval()
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                model(data)

    calibration(model, device, train_dataloader)

    quantizer.compress()

    acc = test(model, device, test_dataloader)
    print(acc)

def finetune_during_quantize(model, device, epoch, train_dataloader, test_dataloader):
    original_checkpoint = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_original_model/pretrained_mobilenet_v2_best_checkpoint.pt"
    model.load_state_dict(torch.load(original_checkpoint))
    configure_config = generate_configure_list(model, mix_bit_setting_cand5, quant_start_step=200)
    print(configure_config)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)
    model.to(device)
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    # quantizer = QAT_Quantizer(model, configure_config, optimizer)
    quantizer = QAT_Quantizer(model, configure_config, optimizer, dummy_input=dummy_input)
    # quantizer = LsqQuantizer(model, configure_config, optimizer, dummy_input=dummy_input)
    quantizer.compress()
    lr_current = 1e-4
    best_acc = 0
    for i in range(epoch):
        # import pdb; pdb.set_trace()
        print(f"Begin to finetune aware quantize epoch {i}")
        lr_current, best_acc = adjust_learning_rate(optimizer, i, lr_current, best_acc)
        print(f"current learning rate: {lr_current}")
        train(model, device, train_dataloader, optimizer)
        acc = test(model, device, test_dataloader)
        if acc > best_acc:
            best_acc = acc
            save_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_finetune.pth"
            torch.save(model.state_dict(), save_path)
            print(f"finetuned model has been saved to {save_path}")
    model_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_qfinetune.pth"
    calibration_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_config.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    print(calibration_config)

def adjust_learning_rate(optimizer, epoch, lr_current, best_acc=None):
    warmup_epoch = 1
    gamma = 0.1
    lr_type = 'cos'
    lr = 1e-4
    if epoch < warmup_epoch:
        lr_current = lr_current*gamma
    elif lr_type == 'cos':
        # cos
        lr_current = 0.5 * lr * (1 + math.cos(math.pi * epoch / 20))
    elif lr_type == 'exp':
        step = 1
        decay = gamma
        lr_current = lr * (decay ** (epoch // step))
    elif epoch in [5, 15]:
        lr_current *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current
    return lr_current, best_acc

def train_conv(model, origin_model, device, data_train_list, optimizer):
    model.train()
    for idx, data in enumerate(data_train_list):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = origin_model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        if idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * idx / len(data_train_list), loss.item()))

def test_conv(model, origin_model, device, data_val_list, optimizer):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in data_val_list:
            data = data.to(device)
            output = model(data)
            target = origin_model(data)
            test_loss += F.mse_loss(output, target).item()
    test_loss /= len(data_val_list)
    print(f"loss: {test_loss}")
    return test_loss

def train_bit_propagate(conv_module, conv_module_cp, module_name, all_shapes, input_bit, weight_bit, output_bit, result_list):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Model_conv(conv_module_cp)
    origin_model = Model_conv(conv_module)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    configure_list = [{
        'quant_types': ['input', 'weight', 'output'],
        'quant_bits': {
            'input': input_bit,
            'weight': weight_bit,
            'output': output_bit
        }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
        'op_types':['Conv2d']
    }]
    quantizer = QAT_Quantizer(model, configure_list, optimizer)
    quantizer.compress()

    module_shapes = all_shapes[module_name]
    dummy_input_shape = module_shapes[0]

    data_train_list = []
    for _ in range(1000):
        dummy_per_image = torch.rand(dummy_input_shape).to(device)
        data_train_list.append(dummy_per_image)
    data_val_list = []
    for _ in range(100):
        dummy_per_image = torch.rand(dummy_input_shape).to(device)
        data_val_list.append(dummy_per_image)

    model.to(device)
    origin_model.to(device)
    for epoch in range(5):
        print('# Epoch {} #'.format(epoch))
        train_conv(model, origin_model, device, data_train_list, optimizer)
        test_loss = test_conv(model, origin_model, device, data_val_list, optimizer)
    result = {f"weight bit set: {weight_bit}, test_loss(after 5 epoch finetuning): {test_loss}"}
    print(result)
    result_list[weight_bit] = test_loss

def finetune_single_conv(model, choose_idx, bit_setting, all_result_list):
    idx = 0
    original_module = None
    choose_module = None
    choose_name = None
    support_ops = [torch.nn.Conv2d]
    for name, module in model.named_modules():
        if type(module) not in support_ops:
            continue
        if idx == choose_idx:
            original_module = module
            choose_module = copy.deepcopy(module)
            choose_name = name
            break
        idx += 1
    assert choose_module is not None
    input_bit = bit_setting[idx-1][1] if bit_setting[idx-1][1] != -1 else 31
    weight_bit = bit_setting[idx][0] if bit_setting[idx][0] != -1 else 31
    output_bit = bit_setting[idx][1] if bit_setting[idx][1] != -1 else 31
    result_list = {}
    
    dummy_input = torch.zeros(1, 3, 224, 224)
    all_shapes = get_all_module_shape(model, dummy_input)

    for i in range(1, min(weight_bit+3, 32)):
        choose_module = copy.deepcopy(original_module)
        train_bit_propagate(original_module, choose_module, choose_name, all_shapes, input_bit, i, output_bit, result_list)
        all_result_list[choose_name] = result_list
        with open('all_result.json', 'w') as f:
            json.dump(all_result_list, f)

def search_all_conv():
    all_result_list = {}
    for i in range(1, 51):
        finetune_single_conv(model, i, mix_bit_setting_cand5, all_result_list)

def get_and_store_all_shape(model):
    dummy_input = torch.zeros(1, 3, 224, 224)
    all_shapes = get_all_module_shape(model, dummy_input)
    shape_json_path = "all_shapes.json"
    with open(shape_json_path, 'w') as f:
        json.dump(all_shapes, f)

def substitute_conv_module(model, choose_idx, input_bits, weight_bits, output_bits):
    original_checkpoint = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_original_model/pretrained_mobilenet_v2_best_checkpoint.pt"
    model.load_state_dict(torch.load(original_checkpoint))
    idx = 0
    support_op_types = [torch.nn.Conv2d, torch.nn.Linear]
    conv_module = None
    conv_name = None
    for name, module in model.named_modules():
        if type(module) in support_op_types:
            if idx == choose_idx:
                conv_module = module
                conv_name = name
                # mix_bit_setting_cand3[idx][0] = weight_bits
                break
            idx += 1
    
    conv_module_cp = copy.deepcopy(conv_module)
    model_conv = Model_conv(conv_module_cp)
    origin_model_conv = Model_conv(conv_module)

    model_conv.to(device)
    origin_model_conv.to(device)

    optimizer_conv = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    configure_list_conv = [{
        'quant_types': ['input', 'weight', 'output'],
        'quant_bits': {
            'input': input_bits,
            'weight': weight_bits,
            'output': output_bits
        }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
        'op_types':['Conv2d']
    }]

    all_shapes_path = "all_shapes.json"

    with open(all_shapes_path, 'r') as f:
        all_shapes = json.load(f)

    module_shapes = all_shapes[conv_name]
    dummy_input_shape = module_shapes[0]

    dummy_input = torch.rand(dummy_input_shape).to(device)

    quantizer_conv = QAT_Quantizer(model_conv, configure_list_conv, optimizer_conv, dummy_input=dummy_input)
    quantizer_conv.compress()


    data_train_list = []
    for _ in range(1000):
        dummy_per_image = torch.rand(dummy_input_shape).to(device)
        data_train_list.append(dummy_per_image)
    data_val_list = []
    for _ in range(100):
        dummy_per_image = torch.rand(dummy_input_shape).to(device)
        data_val_list.append(dummy_per_image)

    for epoch in range(5):
        print('# Epoch {} #'.format(epoch))
        train_conv(model_conv, origin_model_conv, device, data_train_list, optimizer_conv)
        test_loss = test_conv(model_conv, origin_model_conv, device, data_val_list, optimizer_conv)
    result = {f"output bit set: {output_bits}, test_loss(after 5 epoch finetuning): {test_loss}"}
    print(result)

    model_path = "mobilenetv2.pth"
    calibration_path = "mobilenetv2_calibration.pth"
    _ = quantizer_conv.export_model(model_path, calibration_path)

    # _setattr(model, conv_name, conv_module_cp)

    configure_config = generate_configure_list(model, mix_bit_setting_cand5, 200)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    model.to(device)
    quantizer = QAT_Quantizer(model, configure_config, optimizer, dummy_input=dummy_input)
    quantizer.compress()

    #save_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_finetune.pth"
    #model.load_state_dict(torch.load(save_path))

    #calibration_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_config.pth"
    #calibration_config = torch.load(calibration_path)
    #quantizer.load_calibration_config(calibration_config)

    lr_current = 1e-4
    best_acc = 0

    need_finetune = True

    """
    load_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_finetune_new.pth"
    if os.path.isfile(load_path):
        model.load_state_dict(torch.load(load_path))
    """

    acc = test(model, device, test_dataloader)
    if acc > 64:
        need_finetune == False

    for i in range(2):
        # import pdb; pdb.set_trace()
        if need_finetune == False:
            break
        print(f"Begin to finetune aware quantize epoch {i}")
        lr_current, _ = adjust_learning_rate(optimizer, i, lr_current)
        print(f"current learning rate: {lr_current}")
        train(model, device, train_dataloader, optimizer)
        acc = test(model, device, test_dataloader)
        if acc > best_acc:
            save_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_finetune_new.pth"
            torch.save(model.state_dict(), save_path)
            best_acc = acc

    print("after finetune quantize model")
    test(model, device, test_dataloader)

    for name, module in model.named_modules():
        if hasattr(module, 'name') and module.name == conv_name:
            module.module.layer_quant_setting.weight.bits

            print(f"original weight bits: {module.module.layer_quant_setting.weight.bits}")
            module.module.weight.data = conv_module_cp.weight.data
            module.module.layer_quant_setting.weight.bits = weight_bits
            """
            print(f"original weight bits: {module.module.weight_bits}")
            module.module.weight.data = conv_module_cp.weight.data
            module.module.weight_bits = torch.Tensor([weight_bits])
            """

    for name, module in model.named_modules():
        if hasattr(module, 'name') and module.name == conv_name:
            print(f"after substitute bits: {module.module.layer_quant_setting.weight.bits}")

    model.eval()
    print("Begin to test after substituting module")
    test(model, device, test_dataloader)
    best_acc = 0
    for i in range(20):
        print(f"Begin to finetune aware quantize epoch {i}")
        train(model, device, train_dataloader, optimizer)
        acc = test(model, device, test_dataloader)
        if acc > best_acc:
            best_acc = acc
            save_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_finetune.pth"
            torch.save(model.state_dict(), save_path)
            print(f"finetuned model has been saved to {save_path}")
    model_path = "mobilenetv2.pth"
    calibration_path = "mobilenetv2_calibration.pth"
    calibration_config = quantizer_conv.export_model(model_path, calibration_path)
    print(calibration_config)


def substitute_whole_model(origin_model):
    original_checkpoint = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_original_model/pretrained_mobilenet_v2_best_checkpoint.pt"
    origin_model.load_state_dict(torch.load(original_checkpoint))
    idx = 0
    support_op_types = [torch.nn.Conv2d, torch.nn.Linear]
    conv_module = None
    conv_name = None
    original_bit_setting = mix_bit_setting_cand5
    current_bit_setting = copy.deepcopy(mix_bit_setting_cand5)

    for idx, idx_tuple in enumerate(original_bit_setting):
        if idx == 0:
            continue

        input_bits = original_bit_setting[idx-1][1]
        weight_upbound_bits = original_bit_setting[idx][0]
        output_bits = original_bit_setting[idx][1]
        module_idx = 0
        for name, module in origin_model.named_modules():
            if type(module) in support_op_types:
                if module_idx == idx:
                    conv_module = module
                    conv_name = name
                    # mix_bit_setting_cand3[idx][0] = weight_bits
                    break
                module_idx += 1
    
        all_results_path = "all_result.json"
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)
        mse_result = all_results[conv_name]
        upbound_bit = int(list(mse_result.keys())[-1])
        upbound_mse = mse_result[str(upbound_bit)]+mse_result[str(upbound_bit)]/25
        downbound_bit = 31
        for key, val in mse_result.items():
            if val < upbound_mse:
                downbound_bit = int(key)
                break
        print(f"weight_upbound_bits: {weight_upbound_bits}")
        print(f"downbound_bit: {downbound_bit}")

        model = copy.deepcopy(origin_model)
        configure_config = generate_configure_list(model, current_bit_setting, 200)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)
        dummy_input = torch.rand(1, 3, 224, 224).to(device)
        model.to(device)
        quantizer = QAT_Quantizer(model, configure_config, optimizer, dummy_input=dummy_input)
        quantizer.compress()

        lr_current = 1e-4
        best_acc = 0

        need_finetune = True

        acc = test(model, device, test_dataloader)
        if acc > 64:
            need_finetune == False

        for i in range(20):
            # import pdb; pdb.set_trace()
            if need_finetune == False:
                break
            print(f"Begin to finetune aware quantize epoch {i}")
            lr_current, _ = adjust_learning_rate(optimizer, i, lr_current)
            print(f"current learning rate: {lr_current}")
            train(model, device, train_dataloader, optimizer)
            acc = test(model, device, test_dataloader)
            if acc > best_acc:
                save_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_finetune_new.pth"
                torch.save(model.state_dict(), save_path)
                best_acc = acc

        print("after finetune quantize model")
        original_acc = test(model, device, test_dataloader)

        #for weight_bits in range(downbound_bit, weight_upbound_bits+1):
        for weight_bits in range(weight_upbound_bits-1, downbound_bit-1, -1):
            conv_module_cp = copy.deepcopy(conv_module)
            model_conv = Model_conv(conv_module_cp)
            origin_model_conv = Model_conv(conv_module)

            model_conv.to(device)
            origin_model_conv.to(device)

            optimizer_conv = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
            configure_list_conv = [{
                'quant_types': ['input', 'weight', 'output'],
                'quant_bits': {
                    'input': input_bits,
                    'weight': weight_bits,
                    'output': output_bits
                }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
                'op_types':['Conv2d']
            }]

            all_shapes_path = "all_shapes.json"

            with open(all_shapes_path, 'r') as f:
                all_shapes = json.load(f)

            module_shapes = all_shapes[conv_name]
            dummy_input_shape = module_shapes[0]

            dummy_input = torch.rand(dummy_input_shape).to(device)

            quantizer_conv = QAT_Quantizer(model_conv, configure_list_conv, optimizer_conv, dummy_input=dummy_input)
            quantizer_conv.compress()

            data_train_list = []
            for _ in range(1000):
                dummy_per_image = torch.rand(dummy_input_shape).to(device)
                data_train_list.append(dummy_per_image)
            data_val_list = []
            for _ in range(100):
                dummy_per_image = torch.rand(dummy_input_shape).to(device)
                data_val_list.append(dummy_per_image)

            for epoch in range(5):
                print('# Epoch {} #'.format(epoch))
                train_conv(model_conv, origin_model_conv, device, data_train_list, optimizer_conv)
                test_loss = test_conv(model_conv, origin_model_conv, device, data_val_list, optimizer_conv)
            result = {f"output bit set: {output_bits}, test_loss(after 5 epoch finetuning): {test_loss}"}
            print(result)

            #model_path = "mobilenetv2.pth"
            #calibration_path = "mobilenetv2_calibration.pth"
            #_ = quantizer_conv.export_model(model_path, calibration_path)

            bit_setting_tmp = copy.deepcopy(current_bit_setting)
            bit_setting_tmp[idx][0] = weight_bits

            for name, module in model.named_modules():
                if hasattr(module, 'name') and module.name == conv_name:
                    module.module.layer_quant_setting.weight.bits

                    print(f"original weight bits: {module.module.layer_quant_setting.weight.bits}")
                    module.module.weight.data = conv_module_cp.weight.data
                    module.module.layer_quant_setting.weight.bits = weight_bits
                    """
                    print(f"original weight bits: {module.module.weight_bits}")
                    module.module.weight.data = conv_module_cp.weight.data
                    module.module.weight_bits = torch.Tensor([weight_bits])
                    """

            for name, module in model.named_modules():
                if hasattr(module, 'name') and module.name == conv_name:
                    print(f"after substitute bits: {module.module.layer_quant_setting.weight.bits}")

            model.eval()
            print("Begin to test after substituting module")
            test(model, device, test_dataloader)
            best_acc = 0
            for i in range(20):
                print(f"Begin to finetune aware quantize epoch {i}")
                train(model, device, train_dataloader, optimizer)
                acc = test(model, device, test_dataloader)
                if acc > best_acc:
                    best_acc = acc
                    save_path = "/data/v-linbin/bit_propagation_experiment/search_finetune_fit/finetune_quantize_model/mobilenetv2_finetune.pth"
                    torch.save(model.state_dict(), save_path)
                    print(f"finetuned model has been saved to {save_path}")
            if best_acc < original_acc:
                break
            print(f"propagate one module successfully, propagate from {weight_upbound_bits} to {weight_bits} bit")
            current_bit_setting = bit_setting_tmp
            model_path = "mobilenetv2.pth"
            calibration_path = "mobilenetv2_calibration.pth"
            calibration_config = quantizer_conv.export_model(model_path, calibration_path)
            print(current_bit_setting)

def a():
    file_name = "all_result.json"
    with open(file_name, 'r') as f:
        all_result = json.load(f)
    module_names = []
    module_idx = []

    support_op = [torch.nn.Conv2d, torch.nn.Linear]

    idx = 0

    new_result = {}

    for name, module in model.named_modules():
        if type(module) in support_op:
            module_names.append(name)
            module_idx.append(idx)

            if idx == 0:
                continue

            idx += 1

def b():
    import torch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    modules = []
    for name, module in model.named_modules():
        if type(module) == torch.nn.Conv2d:
            modules.append(name)
    return modules

def get_specific_modules_activation(model_original, dummy_input, module_names):
    """
    Get all module shape by adding hooks
    """
    def _pre_forward_hook(self, inp):
        # Only record the first tensor of the input
        return self.pre_forward(inp[0])

    def _post_forward_hook(self, _, out):
        return self.post_forward(out)

    model = copy.deepcopy(model_original).to(device)
    dummy_input = dummy_input.to(device)

    support_op_types = [torch.nn.Conv2d]
    all_handles = []
    all_observers = {}
    all_tensors = {}
    for name, module in model.named_modules():
        if name in module_names:
            all_observers[name] = {}
            all_observers[name]['input_hook'] = RecordingObserver()
            all_observers[name]['output_hook'] = RecordingObserver()
            module.add_module('pre_forward', all_observers[name]['input_hook'])
            module.add_module('post_forward', all_observers[name]['output_hook'])
            all_handles.append(module.register_forward_pre_hook(_pre_forward_hook))
            all_handles.append(module.register_forward_hook(_post_forward_hook))
    model(dummy_input)
    for name, hooks in all_observers.items():
        output_val = hooks['output_hook'].tensor_val
        all_tensors[name] = output_val[0]
    del all_handles
    del all_observers
    del model

    return all_tensors

def get_all_module_shape(model_original, dummy_input):
    """
    Get all module shape by adding hooks
    """
    def _pre_forward_hook(self, inp):
        # Only record the first tensor of the input
        return self.pre_forward(inp[0])

    def _post_forward_hook(self, _, out):
        return self.post_forward(out)

    model = copy.deepcopy(model_original).to(device)
    dummy_input = dummy_input.to(device)

    support_op_types = [torch.nn.Conv2d]
    all_handles = []
    all_observers = {}
    all_shapes = {}
    for name, module in model.named_modules():
        if type(module) in support_op_types:
            all_observers[name] = {}
            all_observers[name]['input_hook'] = RecordingObserver()
            all_observers[name]['output_hook'] = RecordingObserver()
            module.add_module('pre_forward', all_observers[name]['input_hook'])
            module.add_module('post_forward', all_observers[name]['output_hook'])
            all_handles.append(module.register_forward_pre_hook(_pre_forward_hook))
            all_handles.append(module.register_forward_hook(_post_forward_hook))
    model(dummy_input)
    for name, hooks in all_observers.items():
        input_val = hooks['input_hook'].tensor_val
        input_shape = input_val[0].shape if input_val else None
        output_val = hooks['output_hook'].tensor_val
        output_shape = output_val[0].shape if output_val else None
        shapes = [input_shape, output_shape]
        all_shapes[name] = shapes
    del all_handles
    del all_observers
    del model
    return all_shapes

def parse_args():
    parser = argparse.ArgumentParser(description='Example code for pruning MobileNetV2')

    parser.add_argument('--finetune_original_model', type=bool, default=False, help="whether finetune original model")
    parser.add_argument('--finetune_quantize_model', type=bool, default=False, help="whether finetune quantize model")
    parser.add_argument('--search_conv_module', type=bool, default=False, help="whether search candidate conv module which can have lower weight bit")
    parser.add_argument('--substitute_conv_module', type=int, default=None, help="whether substitute candidate conv and validate result")
    parser.add_argument('--substitute_conv_module_finetune', type=int, default=None, help="whether substitute candidate conv and finetune whole model")
    parser.add_argument('--test_checkpoint_acc', type=str, default=None, help="test checkpoint accuracy")
    parser.add_argument('--finetune_post_training_quantize', type=bool, default=False, help="try PTQ quantizer")
    parser.add_argument('--get_and_store_all_shape', type=bool, default=False, help="get the input, weight, output shape of all module in this model")
    parser.add_argument('--substitute_whole_model', type=bool, default=False, help="whether substitute whole model result")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.finetune_original_model:
        finetune_before_quantize(model, device, 180, train_dataloader, test_dataloader)
    if args.finetune_quantize_model:
        finetune_during_quantize(model, device, 20, train_dataloader, test_dataloader)
    if args.search_conv_module:
        search_all_conv()
    if args.finetune_post_training_quantize:
        finetune_post_training_quantize(model, device, train_dataloader, test_dataloader)
    if args.substitute_conv_module:
        # substitute_conv_module(model, 3, 5, 4, 6)
        # substitute_conv_module(model, 2, 6, 4, 5)
        # substitute_conv_module(model, 4, 6, 6, 5)
        substitute_conv_module(model, 2, 6, 1, 5)
        pass
    if args.test_checkpoint_acc:
        model.load_state_dict(torch.load(args.test_checkpoint_acc))
        test(model, device, test_dataloader)
    if args.get_and_store_all_shape:
        get_and_store_all_shape(model)
    if args.substitute_whole_model:
        substitute_whole_model(model)