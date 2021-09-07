import torch
import nni.retiarii.nn.pytorch as nn
import json
from nni.retiarii.nn.pytorch import LayerChoice

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        pass

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)



class TestLayerChoice(nn.Module):
    def __init__(self):
        super().__init__()
        feature = LayerChoice([
                nn.Conv2d(3, 1, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(3),
            ])
        
        self.features = nn.Sequential(feature)

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        pass


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.functional.relu(x)

    def _initialize_weights(self):
        pass


if __name__ == '__main__':
    # model = vgg11()
    # model = TestLayerChoice()
    model = Test()

    model_script = torch.jit.script(model)
    def dfs(model_script, name):
        print(name)
        for k in model_script._modules.keys():
            dfs(model_script._modules[k], name + "." + k)

    print("#----------#")  
    print(model_script)
    print(model_script.graph)
    dfs(model_script, model_script.original_name)
    print("#----------#")  

    input_shape=(1, 3, 5, 5)
    args = torch.randn(*input_shape)

    from nni.retiarii.converter import convert_to_graph
    from nni.retiarii.converter.graph_gen import GraphConverterWithShape

    # PyTorch module to NNI IR
    script_module = torch.jit.script(model)
    converter = GraphConverterWithShape()
    ir_model = convert_to_graph(
        script_module, model, converter=converter, dummy_input=args)

    for k in ir_model.graphs:
        print(f"##### Graph: {k}, {ir_model.graphs[k].python_name}")
        nodes = ir_model.graphs[k].hidden_nodes
        for item in ir_model.graphs[k].hidden_nodes:
            print(f'{item.name}, {item.python_name}')