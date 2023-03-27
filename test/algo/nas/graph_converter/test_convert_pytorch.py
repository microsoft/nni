'''
The tests in this file is copied and transformed from
https://github.com/pytorch/pytorch/blob/master/test/onnx/test_pytorch_onnx_onnxruntime.py
'''

import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import nni.nas.nn.pytorch.layers as nn

from .convert_mixin import ConvertMixin, ConvertWithShapeMixin


class TestPytorch(unittest.TestCase, ConvertMixin):

    def test_embedding_model_with_external_data(self):
        class LargeModel(nn.Module):
            def __init__(self):
                super(LargeModel, self).__init__()
                dim = 15
                n = 4 * 100
                self.emb = nn.Embedding(n, dim)
                self.lin1 = nn.Linear(dim, 1)
                self.seq = nn.Sequential(
                    self.emb,
                    self.lin1,
                )

            def forward(self, input):
                return self.seq(input)

        model = LargeModel()
        x = torch.tensor([2], dtype=torch.long)
        # emb and lin1 is actually not used so they won't appear in generated model
        self.run_test(model, (x, ), strict_load=False)

    @unittest.skip('skip for now, as it needs inject_nn')
    def test_mobilenet_v2_with_external_data(self):
        model = torchvision.models.mobilenet_v2(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        # We are turning off Onnx Runtime optimization off in this test,
        # because external data format is not supported to in ORT optimizer.
        # Once that support is added, we can set ort_optim_on=True (default).
        self.run_test(model, (x, ))

    def test_attribute_with_external_data(self):
        class LargeModel(nn.Module):
            def forward(self, x):
                return x + torch.ones(2, 1024)

        x = torch.randn(2, 1)
        self.run_test(LargeModel(), (x, ))

    @unittest.skip('skip as it has loop')
    def test_subgraph_with_external_data(self):
        class LargeModel(nn.Module):
            def forward(self, x):
                for i in range(x.size(0)):
                    x = x + torch.ones(2, 1024)
                return x

        x = torch.randn(2, 1)
        self.run_test((LargeModel()), (x, ))

    def test_fuse_conv_bn1d(self):
        class Fuse(nn.Module):
            def __init__(self):
                super(Fuse, self).__init__()
                self.conv = nn.Conv1d(16, 33, 3, stride=2)
                self.bn = nn.BatchNorm1d(33)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn2d(self):
        class Fuse(nn.Module):
            def __init__(self):
                super(Fuse, self).__init__()
                self.conv = nn.Conv2d(3, 2, kernel_size=1, stride=2, padding=3, bias=False)
                self.bn = nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn3d(self):
        class Fuse(nn.Module):
            def __init__(self):
                super(Fuse, self).__init__()
                self.conv = nn.Conv3d(3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False)
                self.bn = nn.BatchNorm3d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 10, 50, 100, requires_grad=True)
        self.run_test(model, (x,))

    @unittest.skip('have not supported register_buffer yet')
    def test_reshape_constant_fold(self):
        class Reshape(nn.Module):
            def __init__(self, ):
                super(Reshape, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                return x * scale_1

        x = torch.randn(4, 5)
        self.run_test(Reshape(), (x,))

    def run_word_language_model(self, model_name):
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        model = word_language_model.RNNModel(model_name, ntokens, emsize,
                                             nhid, nlayers, dropout, tied,
                                             batchsize)
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_test(model, (x, model.hidden))

    def get_image_from_url(self, url, size=(300, 200)):
        import os
        from urllib.parse import urlsplit
        from urllib import request
        from PIL import Image
        from torchvision import transforms
        from torch._utils_internal import get_writable_path

        filename = os.path.basename(urlsplit(url)[2])
        data_dir = get_writable_path(os.path.join(os.path.dirname(__file__)))
        path = os.path.join(data_dir, filename)
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb') as f:
            f.write(data)
        image = Image.open(path).convert("RGB")

        image = image.resize(size, Image.BILINEAR)

        to_tensor = transforms.ToTensor()
        return to_tensor(image)

    def get_test_images(self):
        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image = self.get_image_from_url(url=image_url, size=(100, 320))

        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2 = self.get_image_from_url(url=image_url2, size=(250, 380))

        return [image], [image2]

    @unittest.skip('does not support `if A and/or B`')
    def test_faster_rcnn(self):
        from .inject_nn import inject_pytorch_nn, remove_inject_pytorch_nn
        try:
            inject_pytorch_nn()

            model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                                    max_size=300)
            model.eval()
            x = torch.randn(2, 3, 200, 300, requires_grad=True)
            self.run_test(model, (x,))
            dummy_image = [torch.ones(3, 100, 100) * 0.3]
            images, test_images = self.get_test_images()
            self.run_test(model, (images,))
            self.run_test(model, (dummy_image,))
        finally:
            remove_inject_pytorch_nn()

    @unittest.skip('does not support `if A and/or B`')
    def test_mask_rcnn(self):
        from .inject_nn import inject_pytorch_nn, remove_inject_pytorch_nn
        try:
            inject_pytorch_nn()

            model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                                max_size=300)
            images, test_images = self.get_test_images()
            self.run_test(model, (images,))
            dummy_image = [torch.ones(3, 100, 100) * 0.3]
            self.run_test(model, (dummy_image,))
        finally:
            remove_inject_pytorch_nn()

    @unittest.skip('does not support `if A and/or B`')
    def test_keypoint_rcnn(self):
        from .inject_nn import inject_pytorch_nn, remove_inject_pytorch_nn
        try:
            inject_pytorch_nn()

            model = torchvision.models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                                        max_size=300)
            images, test_images = self.get_test_images()
            self.run_test(model, (images,))
            dummy_images = [torch.ones(3, 100, 100) * 0.3]
            self.run_test(model, (dummy_images,))
        finally:
            remove_inject_pytorch_nn()

    def test_shufflenet_v2_dynamic_axes(self):
        from .inject_nn import inject_pytorch_nn, remove_inject_pytorch_nn
        try:
            inject_pytorch_nn()

            model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
            test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)
            self.run_test(model, (dummy_input,))
        finally:
            remove_inject_pytorch_nn()

    @unittest.skip('')
    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    @unittest.skip('')
    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    @unittest.skip('')
    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    @unittest.skip('')
    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    def test_index_1d(self):
        class MyModel(nn.Module):
            def forward(self, input):
                return input[0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1, ))

    def test_index_2d_1dimslice(self):
        class MyModel(nn.Module):
            def forward(self, input):
                return input[0:1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1, ))

    def test_index_2d_sliceint(self):
        class MyModel(nn.Module):
            def forward(self, input):
                return input[1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1, ))

    def test_index_2d_neg_slice(self):
        class MyModel(nn.Module):
            def forward(self, input):
                return input[0:-1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1, ))

    def test_index_mask(self):
        class MyModel(nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.uint8)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1, ))

        class MyModel(nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1, ))

    def test_data(self):
        class Data(nn.Module):
            def forward(self, x):
                return x.new_zeros(x.data.size())

        x = torch.randn(3, 4)
        self.run_test(Data(), (x, ))

    def test_index_mask_nd(self):
        class MyModel(nn.Module):
            def forward(self, input):
                return input[input > 0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1, ))

    @unittest.skip("Tried to access nonexistent attribute or method 'keys' of type 'Tensor (inferred)'.")
    def test_dict(self):
        class MyModel(nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in[list(x_in.keys())[0]], list(x_in.keys())[0])
                return x_out

        x = {torch.tensor(1.): torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x, {}))

    @unittest.skip("Unsupported operation: indexing tensor with unsupported index type 'str'.")
    def test_dict_str(self):
        class MyModel(nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x, {}))

    @unittest.skip('Convert graph error')
    def test_optional_inputs_with_no_optionals(self):
        class NoOptionalModel(nn.Module):
            def forward(self, input):
                return input

        # Without empty optional arguments dictionary
        x = torch.randn(2, 3)
        self.run_test(NoOptionalModel(), (x,))
        # With empty optional arguments dictionary
        y = torch.randn(2, 3)
        self.run_test(NoOptionalModel(), (y, {}))

    # NOTE: torch script gets an incorrect graph...
    def test_optional_inputs_with_mixed_optionals(self):
        class MixedModel(nn.Module):
            def forward(self, x, y, z):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        # Without optional arguments dictionary
        self.run_test(MixedModel(), (x, y, None))
        #self.run_test(MixedModel(), (x, None, z, ))
        # With optional arguments dictionary
        #self.run_test(MixedModel(), (x, {'y': y, 'z': None}))
        #self.run_test(MixedModel(), (x, {'y': None, 'z': z}))
        #self.run_test(MixedModel(), (x, {'z': z}))
        #self.run_test(MixedModel(), (x, {'y': y}))

    @unittest.skip('torch script gets an incorrect graph...')
    def test_optional_inputs_with_all_optionals(self):
        class AllOptionalModel(nn.Module):
            def forward(self, y, z):
                if y is not None:
                    return y
                if z is not None:
                    return z

        y = torch.randn(2, 3)
        # Without optional arguments dictionary
        self.run_test(AllOptionalModel(), (y, None))
        # With optional arguments dictionary
        #self.run_test(AllOptionalModel(), {'y': y, 'z': None})

    @unittest.skip('torch script gets an incorrect graph...')
    def test_none_as_input(self):
        class Model(nn.Module):
            def forward(self, x, y):
                if y is not None:
                    return x + y
                return x

        x = torch.randn(2, 3)
        self.run_test(Model(), (x, None))

    @unittest.skip('jit cannot correctly deal with tuple as input argument')
    def test_none_as_tuple_input(self):
        class Model(nn.Module):
            def forward(self, x, y):
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y)))

    def test_cste_script(self):
        class MyModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0)), torch.ones((x.size(1), x.size(0)), dtype=torch.int64)

        x = torch.randn(3, 4)
        self.run_test(MyModel(), (x, ))

    def test_scalar_tensor(self):
        class test(nn.Module):
            def forward(self, input):
                return torch.scalar_tensor(input.size(0)), \
                    torch.scalar_tensor(input.size(1), dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        model = test()
        self.run_test(model, (x, ))

    def test_tensor(self):
        class ScalarInputModel(nn.Module):
            def forward(self, input):
                return torch.tensor(input.shape[1])

        x = torch.randn(3, 4)
        self.run_test(ScalarInputModel(), (x, ))

        class TensorInputModel(nn.Module):
            def forward(self, input):
                return torch.tensor([input.shape[0], input.shape[1]])

        x = torch.randn(3, 4)
        self.run_test(TensorInputModel(), (x, ))

        class FloatInputModel(nn.Module):
            def forward(self, input):
                return torch.tensor([float(input)])

        x = torch.randn(1)
        self.run_test(FloatInputModel(), (x, ))

        class InputWithDtypeModel(nn.Module):
            def forward(self, input):
                return torch.tensor(input.shape[1], dtype=torch.long)

        x = torch.randn(3, 4)
        self.run_test(InputWithDtypeModel(), (x, ))

        class MixedInputModel(nn.Module):
            def forward(self, input):
                return torch.tensor([input.shape[0], int(input)])

        x = torch.randn(1)
        self.run_test(MixedInputModel(), (x, ))

    def test_hardtanh(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Hardtanh(-1.5, 2.5)
            def forward(self, x):
                return self.m(x)

        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(MyModel(), (x, ))

    def test_hardtanh_script_with_default_values(self):
        class MyModel(nn.Module):
            def forward(self, x):
                return F.hardtanh(x)

        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(MyModel(), (x, ))

    def test_hardswish(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Hardswish()
            def forward(self, x):
                return self.m(x)

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(MyModel(), (x, ))

        # Testing edge cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(MyModel(), (x, ))
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(MyModel(), (x, ))

    def test_hardswish_script(self):
        class MyModel(nn.Module):
            def forward(self, x):
                return F.hardswish(x)

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(MyModel(), (x, ))

    def test_clamp(self):
        class ClampModel(nn.Module):
            def forward(self, x):
                return x.clamp(-0.5, 0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampModel(), (x, ))

        class ClampMinModel(nn.Module):
            def forward(self, x):
                return x.clamp(min=-0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMinModel(), (x, ))

        class ClampMaxModel(nn.Module):
            def forward(self, x):
                return x.clamp(max=0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMaxModel(), (x, ))

    def test_clamp_dyn(self):
        class ClampMaxModel(nn.Module):
            def forward(self, x):
                return x.clamp(None, x.size(0))

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMaxModel(), (x, ))


        class ClampMinModel(nn.Module):
            def forward(self, x):
                return x.clamp(x.size(0), None)

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMinModel(), (x, ))

        class ClampMinMaxModel(nn.Module):
            def forward(self, x):
                return x.clamp(x.size(0), x.size(1))

        x = torch.arange(16).view(2, 8).float()
        self.run_test(ClampMinMaxModel(), (x, ))

    def test_full_trace(self):
        class FullModel(nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModel(), (x, ))

    def test_full_script(self):
        class FullModelScripting(nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModelScripting(), (x, ))

    def test_fuse_addmm(self):
        class AddmmModel(nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddmmModel(), (x, ))

    def test_maxpool(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool1d(2, stride=1)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50)
        self.run_test(MyModel(), (x, ))

    def test_conv(self):
        class TraceModel(nn.Module):
            def __init__(self):
                super(TraceModel, self).__init__()
                self.conv1 = nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                self.conv3 = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 100)
        x3 = torch.randn(20, 16, 10, 50, 100)

        self.run_test(TraceModel(), (x1, x2, x3, ))

    def test_conv_shape_inference(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv2 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

            def forward(self, input):
                return self.conv2(input) + 2

        x = torch.randn(20, 16, 50, 100)
        self.run_test(Model(), (x, ))

    def test_conv_transpose(self):
        class TraceModel(nn.Module):
            def __init__(self):
                super(TraceModel, self).__init__()
                self.conv1 = nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                self.conv3 = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 100)
        x3 = torch.randn(20, 16, 10, 50, 100)

        self.run_test(TraceModel(), (x1, x2, x3, ))

    # Conversion of Transpose depends on input shape to be known.
    # The following test only works when onnx shape inference is enabled.
    def test_transpose_infer_shape(self):
        class TransposeModule(nn.Module):
            def __init__(self):
                super(TransposeModule, self).__init__()
                self.conv = nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test(TransposeModule(), (x, ))

    def squeeze_model_tests(self, d, x1):
        class Squeeze(nn.Module):
            def __init__(self, d):
                super(Squeeze, self).__init__()
                self.d = d

            def forward(self, x):
                if self.d is not None:
                    return torch.squeeze(x, dim=self.d)
                else:
                    return torch.squeeze(x)

        self.run_test(Squeeze(d), (x1, ))

    def test_squeeze_without_no_op(self):
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(1, x)

    def test_squeeze_neg_without_no_op(self):
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(-2, x)

    def test_squeeze_all_dims(self):
        x_squeeze = torch.randn(2, 1, 4)
        self.squeeze_model_tests(None, x_squeeze)

    def test_squeeze_no_op(self):
        x_noop = torch.randn(2, 1, 4)
        self.squeeze_model_tests(2, x_noop)

    def test_squeeze_runtime_dim(self):
        class Squeeze(nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test(Squeeze(), (d1, d4))
        self.run_test(Squeeze(), (d3, d4))

    def test_squeeze(self):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim=-2)

        x = torch.randn(2, 1, 4)
        self.run_test(Squeeze(), (x, ))

    def test_unsqueeze(self):
        class Unsqueeze(nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, dim=-2)

        x = torch.randn(2, 3, 4)
        self.run_test(Unsqueeze(), (x, ))

    def test_maxpool_default_stride(self):
        class MaxPoolModel(nn.Module):
            def forward(self, x):
                return F.max_pool2d(x, 2)

        model = MaxPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, (x, ))

    def test_maxpool_adaptive(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.AdaptiveMaxPool1d((5), return_indices=False)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(MyModel(), (x, ))

    def test_maxpool_2d(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool2d(5, padding=(1, 2))
            def forward(self, x):
                return self.m(x)
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(MyModel(), (x, ))

    def test_maxpool_1d_ceil(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool1d(3, 2, ceil_mode=True)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50)
        self.run_test(MyModel(), (x, ))

    def test_maxpool_2d_ceil(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool2d(3, 2, ceil_mode=True)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50, 32)
        self.run_test(MyModel(), (x, ))

    def test_maxpool_3d_ceil(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool3d(3, 2, ceil_mode=True)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(MyModel(), (x, ))

    @unittest.skip('jit error: Return value was annotated as having type Tensor but is actually of type Tuple[Tensor, Tensor]')
    def test_maxpool_with_indices(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool1d(2, stride=1, return_indices=True)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50)
        self.run_test(MyModel(), (x, ))

    def test_maxpool_dilation(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.MaxPool1d(2, stride=1, dilation=2)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50)
        self.run_test(MyModel(), (x, ))

    def test_avgpool_default_stride(self):
        class AvgPoolModel(nn.Module):
            def forward(self, x):
                return F.avg_pool2d(x, 2)

        model = AvgPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, (x, ))

    def test_avgpool(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.AvgPool1d(2, stride=1)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50)
        self.run_test(MyModel(), (x, ))

    def test_avgpool_1d_ceil(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.AvgPool1d(3, 2, ceil_mode=True)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(1, 1, 7)
        self.run_test(MyModel(), (x, ))

    def test_avgpool_2d_ceil(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.AvgPool2d(3, 2, ceil_mode=True)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50, 32)
        self.run_test(MyModel(), (x, ))

    def test_avgpool_3d_ceil(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.AvgPool3d(3, 2, ceil_mode=True)
            def forward(self, x):
                return self.m(x)

        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(MyModel(), (x, ))

    @unittest.skip('Unsupported op type aten::is_floating_point in if condition')
    def test_floating_point(self):
        class FloatingPoint(nn.Module):
            def forward(self, x):
                if x.is_floating_point():
                    return x.new_zeros(x.shape)
                return x.new_zeros(x.shape)

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), (x, ))

        class FloatingPoint(nn.Module):
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x + 1
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), (x, ))

    # Operator rank mismatch between outputs of two branches for opsets below 11.
    @unittest.skip('Unsupported op type aten::size in if condition')
    def test_floating_point_infer_dtype(self):
        class FloatingPoint(nn.Module):
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x.new_zeros(x.shape[1:])
                    return x.new_zeros(x.shape)
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), (x, ))

        class FloatingPoint(nn.Module):
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x
                return x

        x = torch.randn(2, 3, 4).to(torch.int32)
        self.run_test(FloatingPoint(), (x, ))

    def test_arithmetic(self):
        class ArithmeticModule(nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), (x, ))

    # In scripting the first transpose node do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_arithmetic_infer_dtype(self):
        class ArithmeticModule(nn.Module):
            def forward(self, x):
                x = x.t()
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3)
        self.run_test(ArithmeticModule(), (x, ))

    @unittest.skip('tensor op type aten::to has more than one matched')
    def test_floor_div(self):
        class FloorDivModule(nn.Module):
            def forward(self, x, y):
                return x // 3, x // 2., \
                    x.to(dtype=torch.float64) // 3, x.to(dtype=torch.float64) // 2., \
                    x.to(dtype=torch.int64) // 3, x.to(dtype=torch.int64) // 2., \
                    x // (y + 1.).to(dtype=torch.int64), x // y, \
                    x.to(dtype=torch.float64) // y.to(dtype=torch.int64), x.to(dtype=torch.float64) // y.to(dtype=torch.float64), \
                    x.to(dtype=torch.int64) // y.to(dtype=torch.int64), x.to(dtype=torch.int64) // y

        x = torch.randn(2, 3, 4)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    def test_floor_div_script(self):
        class FloorDivModule(nn.Module):
            def forward(self, x, y):
                return x // 3, x // 2., x // y

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    def test_floordiv(self):
        class FloordivModule(nn.Module):
            def forward(self, x):
                return x.new_zeros(x.size(2) // x.size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(FloordivModule(), (x,))

    def test_div(self):
        class DivModule(nn.Module):
            def forward(self, x, y):
                return torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        self.run_test(DivModule(), (x, y))
        self.run_test(DivModule(), (x.float(), y.float()))

    # Note: div cannot (generally) be exported via scripting
    # since its type promotion logic is dependent on knowing the scalar types
    # of the input tensors. That is, the ONNX graph is dependent on the
    # data type of the inputs. This makes it appropriate for tracing only.
    def test_div_promotion_trace(self):
        class DivModule(nn.Module):
            def forward(self, x, y):
                return torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        prev_default = torch.get_default_dtype()

        torch.set_default_dtype(torch.float)
        self.run_test(DivModule(), (x, y))

        torch.set_default_dtype(torch.double)
        self.run_test(DivModule(), (x, y))

        torch.set_default_dtype(prev_default)

    # In scripting x, y do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_div_promotion_script(self):
        class DivModule(nn.Module):
            def forward(self, x, y):
                # Add transpose to hide shape/type information
                # Otherwise shape and type are still avaiable from input.
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                return torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        prev_default = torch.get_default_dtype()

        # 1. x,y are int, and output is float.
        #    This can be handled by the default case, where both are cast to float.
        #    It works even if type of x, y are unknown.
        torch.set_default_dtype(torch.float)
        self.run_test((DivModule()), (x, y))

        # 2. x,y are int, and output is double.
        #    This can be handled by the default case, where both are cast to double.
        #    It works even if type of x, y are unknown.
        torch.set_default_dtype(torch.double)
        self.run_test((DivModule()), (x, y))

        # 3. x is int, y is double, and output is double.
        #    This can only be handled when both type of x and y are known.
        torch.set_default_dtype(prev_default)
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.double)
        self.run_test((DivModule()), (x, y))

    def test_slice_trace(self):
        class MyModule(nn.Module):
            def forward(self, x):
                return x[0:1]

        x = torch.randn(3)
        self.run_test(MyModule(), (x, ))

    def test_slice_neg(self):
        class NegSlice(nn.Module):
            def forward(self, x):
                return x[-1:]

        x = torch.randn(3, 4, 5)
        self.run_test(NegSlice(), (x, ))

    def test_slice_neg_large(self):
        class NegSlice(nn.Module):
            def forward(self, x):
                return x[:, :, -3:-1, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), (x, ))

    def test_slice_neg_large_negone(self):
        class NegSlice(nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), (x, ))

    @unittest.skip('strange torch script graph')
    def test_slice_with_input_index(self):
        class InputIndexSlice(nn.Module):
            def forward(self, x, y):
                x[:y.size(0), 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

    @unittest.skip('Loop has not been supported yet!')
    def test_slice_dynamic(self):
        class DynamicSliceExportMod(nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:x.size(0) - i, i:x.size(2), i:3])
                return results

        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        self.run_test(DynamicSliceExportMod(), (x, ))

    def test_slice_dynamic_script(self):
        class DynamicSliceModel(nn.Module):
            def forward(self, x):
                return x[1:x.size(1)]

        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), (x, ))

    def test_slice_dynamic_shape_script(self):
        class DynamicSliceModel(nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1:x.size(2)])

        x = torch.rand(1, 2, 3, 4)
        self.run_test(DynamicSliceModel(), (x, ))

    @unittest.skip('Loop has not been supported yet!')
    def test_slice_dynamic_to_end(self):
        class DynamicSliceExportMod(nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return results

        x = torch.rand(5, 5, 5)
        self.run_test(DynamicSliceExportMod(), (x, ))

    def test_square(self):
        class Square(nn.Module):
            def forward(self, x):
                return torch.square(x)

        x = torch.randn(2, 3, 4)
        self.run_test(Square(), (x, ))

    def test_arange_dynamic(self):
        class ArangeModel(nn.Module):
            def forward(self, input):
                return torch.arange(input.shape[0]), \
                    torch.arange(12), \
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5)

        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        self.run_test(ArangeModel(), (x, ))

    @unittest.skip('mismatched aten::arange definition, does not support `out`')
    def test_dynamic_arange_out(self):
        class ArangeOutModel(nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8)
        self.run_test(ArangeOutModel(), (x, ))

    @unittest.skip('mismatched aten::arange definition, does not support `out`')
    def test_dynamic_arange_start_out(self):
        class ArangeStartOutModel(nn.Module):
            def forward(self, start, end):
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8)
        self.run_test(ArangeStartOutModel(), (x, y))

    def test_arange(self):
        class ArangeModel(nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModel(), (x, y))

    @unittest.skip('mismatched aten::arange definition, does not support `out`')
    def test_arange_out(self):
        class ArangeOutModel(nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeOutModel(), (x, ))

    @unittest.skip('mismatched aten::arange definition, does not support `out`')
    def test_arange_start_out(self):
        class ArangeStartOutModel(nn.Module):
            def forward(self, start, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeStartOutModel(), (x, y))

    def test_arange_no_type(self):
        class ArangeModel(nn.Module):
            def forward(self, end):
                return torch.arange(end), \
                    torch.arange(0, end)

        x = torch.tensor(6.2, dtype=torch.float)
        self.run_test(ArangeModel(), (x, ))

    def test_size(self):
        class SizeModel(nn.Module):
            def forward(self, input):
                return torch.arange(input.size(0)), torch.arange(input.size(-1)), torch.ones(input.shape)

        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(), (x, ))

    def test_size2(self):
        class SizeModel(nn.Module):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b
            def forward(self, input):
                if self.a < self.b:
                    return torch.arange(input.size(0)), torch.arange(input.size(-1)), torch.ones(input.shape)

        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(5, 10), (x, ))

    def test_python_name(self):
        from .inject_nn import inject_pytorch_nn, remove_inject_pytorch_nn
        try:
            inject_pytorch_nn()
            torchvision_model_zoo = {
                'resnet18': torchvision.models.resnet18(),
                'alexnet': torchvision.models.alexnet(),
                'vgg16': torchvision.models.vgg16(),
                'squeezenet': torchvision.models.squeezenet1_0(),
                'shufflenet_v2': torchvision.models.shufflenet_v2_x1_0(),
                'mobilenet_v2': torchvision.models.mobilenet_v2(),
                'resnext50_32x4d': torchvision.models.resnext50_32x4d(),
                'wide_resnet50_2': torchvision.models.wide_resnet50_2(),
                'mnasnet': torchvision.models.mnasnet1_0(),
            }
            dummy_input=torch.randn(1, 3, 224, 224)
            for model in torchvision_model_zoo.values():
                model_ir = self._convert_model(model, dummy_input)
                current_name = [node.python_name for node in model_ir.get_nodes() if node.python_name]
                mentioned = set()
                for k in model.state_dict():
                    k = ".".join(k.split(".")[:-1])
                    if k not in mentioned:
                        assert k in current_name, f'{k} not in state_name'
                        mentioned.add(k)
        finally:
            remove_inject_pytorch_nn()

class TestPytorchWithShape(TestPytorch, ConvertWithShapeMixin):

    @unittest.skip(reason='trace fails because type is not supported.')
    def test_optional_inputs_with_mixed_optionals(self):
        ...
