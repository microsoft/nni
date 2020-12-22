import sys, os
import torch

sys.path.append("/home/v-linbin/MPQS")
from torchvision import datasets, transforms
from resnet import resnet18

import integrated_tensorrt
from integrated_tensorrt import CalibrateType

config = {
'Conv_0':"int8",
'Relu_1':"int8",
'MaxPool_2':"int8",
'Conv_3':"int8",
'Relu_4':"int8",
'Conv_5':"int8",
'Add_6':"int8",
'Relu_7':"int8",
'Conv_8':"int8",
'Relu_9':"int8",
'Conv_10':"int8",
'Add_11':"int8",
'Relu_12':"int8",
'Conv_13':"int8",
'Relu_14':"int8",
'Conv_15':"int8",
'Conv_16':"int8",
'Add_17':"int8",
'Relu_18':"int8",
'Conv_19':"int8",
'Relu_20':"int8",
'Conv_21':"int8",
'Add_22':"int8",
'Relu_23':"int8",
'Conv_24':"int8",
'Relu_25':"int8",
'Conv_26':"int8",
'Conv_27':"int8",
'Add_28':"int8",
'Relu_29':"int8",
'Conv_30':"int8",
'Relu_31':"int8",
'Conv_32':"int8",
'Add_33':"int8",
'Relu_34':"int8",
'Conv_35':"int8",
'Relu_36':"int8",
'Conv_37':"int8",
'Conv_38':"int8",
'Add_39':"int8",
'Relu_40':"int8",
'Conv_41':"int8",
'Relu_42':"int8",
'Conv_43':"int8",
'Add_44':"int8",
'Relu_45':"int8",
'GlobalAveragePool_46':"int8",
'Shape_47':"int32",
'191':"int32",
'Gather_49':"int32",
'Unsqueeze_50':"int32",
'259':"int32",
'Concat_51':"int32",
# 'Reshape_52':"int32",
'(Unnamed Layer* 54) [Shape]':"int32",
'(Unnamed Layer* 55) [Constant]':"int32",
'(Unnamed Layer* 56) [Concatenation]':"int32",
'(Unnamed Layer* 57) [Constant]':"int32",
'(Unnamed Layer* 58) [Gather]':"int32",
# '(Unnamed Layer* 59) [Shuffle]':"int32",
'Gemm_53':"int32",
'(Unnamed Layer* 61) [Constant]':"int32",
'(Unnamed Layer* 62) [Shape]':"int32",
'(Unnamed Layer* 63) [Gather]':"int32",
# '(Unnamed Layer* 64) [Shuffle]':"int32",
}

def get_testset():
    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data.cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=10000, shuffle=False)
    test_set = next(iter(test_loader))[0].numpy()
    return test_set


def main():
    # model = resnet18(pretrained=True, progress=True, device='cuda')
    model = resnet18(pretrained=True)
    model.load_state_dict(torch.load("/home/v-linbin/PyTorch_CIFAR10/cifar10_models/state_dicts/resnet18.pt"))

    # parameter init for torch model to onnx
    onnx_path = "/home/v-linbin/cifar10model/state_dicts/resnet18.onnx"

    batch_size = 1
    input_shape = (batch_size, 3, 32, 32)
    input_names = ["actual_input_1"]+ [ "learned_%d" % i for i in range(16) ]
    output_names = ["output1"]
    calibration_cache = "cifar_calibration.cache"

    test_set = get_testset()

    integrated_trt = integrated_tensorrt.TensorRt(model, onnx_path, input_shape, config=config, extra_layer_bit='float32', strict_datatype=True, using_calibrate=True, 
    calibrate_type=CalibrateType.ENTROPY2, calib_data=test_set, calibration_cache = calibration_cache, batchsize=batch_size, input_names=input_names, output_names=output_names)
    integrated_trt.tensorrt_build()

    output, time = integrated_trt.inference(test_set)

    print("elapsed_time: ", time)
    
if __name__ == '__main__':
    main()
