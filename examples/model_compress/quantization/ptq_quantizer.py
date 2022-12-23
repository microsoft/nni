import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.quantization

# from nni.algorithms.compression.pytorch.quantization import PtqQuantizer
# from nni.algorithms.compression.v2.pytorch.utils import TorchEvaluator
from nni.compression.pytorch.quantization import PtqQuantizer
from nni.compression.pytorch.utils import TorchEvaluator
from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
from nni.compression.pytorch.quantization_speedup.calibrator import Calibrator

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

#data_path = '/data/data0/v-zhiqxi/imagenet-raw-data'
data_path = '/data'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        #self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            #return self.skip_add.add(x, self.conv(x))
            return torch.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        #self.quant = QuantStub()
        #self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        #x = self.quant(x)

        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        #x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches, device):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    total_time = 0
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            start_time = time.time()
            output = model(image)
            time_span = time.time() - start_time
            total_time += time_span
            print()
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                break
    print('inference time: ', total_time / neval_batches)
    return top1, top5


def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
           data_path, split="train",
         transform=transforms.Compose([
                   transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize,
               ]))
    dataset_test = torchvision.datasets.ImageNet(
          data_path, split="train",
              transform=transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  normalize,
              ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)
    return data_loader, data_loader_test


def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model


def test_trt(engine, data_loader, neval_batches):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    total_time = 0
    for image, target in data_loader:
        output, time_span = engine.inference(image)
        print('time: ', time_span)
        total_time += time_span
        output = output.view(-1, 1000)
        cnt += 1
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        if cnt >= neval_batches:
            break
    print('inference time: ', total_time / neval_batches)
    return top1, top5


def test_trt_no_calibration(test_data_loader, num_eval_batches):
    config_list = [{
        'quant_types': ['input', 'weight', 'output'],
        'quant_bits': {'input': 8, 'weight': 8, 'output': 8},
        'quant_dtype': 'int',
        'quant_scheme': 'per_tensor_symmetric',
        'op_types': ['default']
    }]
    device = torch.device('cuda')
    def my_eval(model):
        evaluate(model, test_data_loader,
                 neval_batches=2, device=device)

    myModel = load_model(saved_model_dir + float_model_file)
    myModel.eval()
    myModel.to(device)
    dummy_input = torch.Tensor(64, 3, 224, 224)
    dummy_input = dummy_input.to(device)
    predict_func = TorchEvaluator(predicting_func=my_eval, dummy_input=dummy_input)
    quantizer = PtqQuantizer(myModel, config_list, predict_func, True)
    sim_quant_model, quant_result_conf = quantizer.compress()
    calibration_config = quantizer.export_model()
    print('quant result config: ', calibration_config)

    input_shape = (64, 3, 224, 224)
    newModel = load_model(saved_model_dir + float_model_file)
    newModel.eval()
    newModel.to(device)
    engine = ModelSpeedupTensorRT(newModel, input_shape, config=calibration_config)
    engine.compress()
    top1, top5 = test_trt(engine, test_data_loader, neval_batches=num_eval_batches)
    print('accuracy: ', top1, top5)

def test_trt_calibration(test_data_loader, num_eval_batches):
    # prepare calibrator
    data_iter = iter(test_data_loader)
    batch1 = next(data_iter)[0]
    batch2 = next(data_iter)[0]
    calib_data = torch.cat((batch1, batch2), 0)
    calib_data = calib_data.numpy()
    calib = Calibrator(calib_data, 'data/calib_cache_file.cache', batch_size=64)
    # speedup and inference
    input_shape = (64, 3, 224, 224)
    device = torch.device('cuda')
    newModel = load_model(saved_model_dir + float_model_file)
    newModel.eval()
    newModel.to(device)
    engine = ModelSpeedupTensorRT(newModel, input_shape, config=None)
    engine.compress_with_calibrator(calib)
    top1, top5 = test_trt(engine, test_data_loader, neval_batches=num_eval_batches)
    print('accuracy: ', top1, top5)

if __name__ == '__main__':
    num_eval_batches = 100
    data_loader, test_data_loader = prepare_data_loaders(data_path, train_batch_size=32, eval_batch_size=64)
    #test_trt_calibration(test_data_loader, num_eval_batches)
    test_trt_no_calibration(test_data_loader, num_eval_batches)
