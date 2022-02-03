# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, 
        n_class=1000, 
        input_size=224, 
        width_mult=1,
        input_channel = 32,
        last_channel = 1280,
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2 if n_class==1000 else 1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # it's same with .mean(3).mean(2), but
        # speedup only suport the mean option
        # whose output only have two dimensions
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def ckpt_to_mobilenetv2(ckpt,n_class=1000):
    interverted_residual_setting=[
        [1, 16, 1, 1],
        [6, 24, 1, 2], [6, 24, 1, 1],
        [6, 32, 1, 2], [6, 32, 1, 1], [6, 32, 1, 1],
        [6, 64, 1, 2], [6, 64, 1, 1], [6, 64, 1, 1], [6, 64, 1, 1],
        [6, 96, 1, 1], [6, 96, 1, 1], [6, 96, 1, 1],
        [6, 160, 1, 2], [6, 160, 1, 1], [6, 160, 1, 1],
        [6, 320, 1, 1]]
    input_channel=ckpt['features.1.conv.4.weight'].shape[0]
    interverted_residual_setting[0][1]=input_channel
    for i in range(2,18):
        interverted_residual_setting[i-1][0]=ckpt['features.%d.conv.6.weight'%i].shape[1]/input_channel
        input_channel=ckpt['features.%d.conv.6.weight'%i].shape[0]
        interverted_residual_setting[i-1][1]=input_channel
    input_channel=ckpt['features.0.1.weight'].shape[0]
    last_channel=ckpt['features.18.1.weight'].shape[0]
    model=MobileNetV2(n_class=n_class,input_channel=input_channel,interverted_residual_setting=interverted_residual_setting,last_channel=last_channel)
    model.load_state_dict(ckpt)
    return model

def ckpt_to_mobilenetv1(ckpt,n_class=1000):
    model=mobilenetv2_to_mobilenetv1(MobileNetV2(n_class=n_class))
    channels=[]
    for k,v in ckpt.items():
        if len(v.shape)==4:
            channels.append(v.shape[0])
    in_channels=3
    features=[]
    for m in model.features:
        if isinstance(m,nn.Conv2d):
            out_channels=channels[0]
            features.append(nn.Conv2d(in_channels,out_channels,kernel_size=m.kernel_size,stride=m.stride,padding=m.padding,groups=in_channels if m.groups!=1 else 1,bias=m.bias))
            channels.pop(0)
            in_channels=out_channels
        elif isinstance(m,nn.BatchNorm2d):
            features.append(nn.BatchNorm2d(in_channels))
        elif isinstance(m,nn.ReLU6):
            features.append(nn.ReLU6(inplace=True))
        elif isinstance(m,nn.PReLU):
            features.append(nn.PReLU(in_channels))
        else:
            print(m)
    model.features=nn.Sequential(*features)
    model.classifier[1]=nn.Linear(out_channels,n_class)
    model.load_state_dict(ckpt)
    return model

def ckpt_to_mobilenet(model_type, ckpt_path, n_class=1000):
    # the checkpoint can be state_dict exported by amc_search.py or saved by amc_train.py
    print('=> Loading checkpoint {} ..'.format(ckpt_path))
    sd = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if 'state_dict' in sd:  # a checkpoint but not a state_dict
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    if model_type == 'rmnetv2':
        net = ckpt_to_mobilenetv2(sd,n_class)
        net = mobilenetv2_to_mobilenetv1(net)
    elif model_type == 'rmnetv1':
        net = ckpt_to_mobilenetv1(sd,n_class)
    else:
        raise NotImplementedError
    return net
        
def fuse_cbcb(conv1,bn1,conv2,bn2):
    inp=conv1.in_channels
    mid=conv1.out_channels
    oup=conv2.out_channels
    conv1=torch.nn.utils.fuse_conv_bn_eval(conv1.eval(),bn1.eval())
    fused_conv=nn.Conv2d(inp,oup,1,bias=False)
    fused_conv.weight.data=(conv2.weight.data.view(oup,mid)@conv1.weight.data.view(mid,-1)).view(oup,inp,1,1)
    bn2.running_mean-=conv2.weight.data.view(oup,mid)@conv1.bias.data
    return fused_conv,bn2
    
def fuse_cb(conv_w, bn_rm, bn_rv,bn_w,bn_b, eps):
    bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = bn_rm * bn_var_rsqrt * bn_w-bn_b
    return conv_w,conv_b

def rm_r(model):    
    inp = model.conv[0].in_channels
    mid = inp+model.conv[0].out_channels
    oup = model.conv[6].out_channels

    running1 = nn.BatchNorm2d(inp,affine=False)
    running2 = nn.BatchNorm2d(oup,affine=False)

    idconv1 = nn.Conv2d(inp, mid, kernel_size=1, bias=False).eval()
    idbn1=nn.BatchNorm2d(mid).eval()

    nn.init.dirac_(idconv1.weight.data[:inp])
    bn_var_sqrt=torch.sqrt(running1.running_var + running1.eps)
    idbn1.weight.data[:inp]=bn_var_sqrt
    idbn1.bias.data[:inp]=running1.running_mean
    idbn1.running_mean.data[:inp]=running1.running_mean
    idbn1.running_var.data[:inp]=running1.running_var

    idconv1.weight.data[inp:]=model.conv[0].weight.data
    idbn1.weight.data[inp:]=model.conv[1].weight.data
    idbn1.bias.data[inp:]=model.conv[1].bias.data
    idbn1.running_mean.data[inp:]=model.conv[1].running_mean
    idbn1.running_var.data[inp:]=model.conv[1].running_var
    idrelu1 = nn.PReLU(mid)
    torch.nn.init.ones_(idrelu1.weight.data[:inp])
    torch.nn.init.zeros_(idrelu1.weight.data[inp:])

    idconv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=model.stride, padding=1,groups=mid, bias=False).eval()
    idbn2=nn.BatchNorm2d(mid).eval()

    nn.init.dirac_(idconv2.weight.data[:inp],groups=inp)
    idbn2.weight.data[:inp]=idbn1.weight.data[:inp]
    idbn2.bias.data[:inp]=idbn1.bias.data[:inp]
    idbn2.running_mean.data[:inp]=idbn1.running_mean.data[:inp]
    idbn2.running_var.data[:inp]=idbn1.running_var.data[:inp]

    idconv2.weight.data[inp:]=model.conv[3].weight.data
    idbn2.weight.data[inp:]=model.conv[4].weight.data
    idbn2.bias.data[inp:]=model.conv[4].bias.data
    idbn2.running_mean.data[inp:]=model.conv[4].running_mean
    idbn2.running_var.data[inp:]=model.conv[4].running_var
    idrelu2 = nn.PReLU(mid)
    torch.nn.init.ones_(idrelu2.weight.data[:inp])
    torch.nn.init.zeros_(idrelu2.weight.data[inp:])

    idconv3 = nn.Conv2d(mid, oup, kernel_size=1, bias=False).eval()
    idbn3=nn.BatchNorm2d(oup).eval()

    nn.init.dirac_(idconv3.weight.data[:,:inp])
    idconv3.weight.data[:,inp:],bias=fuse_cb(model.conv[6].weight,model.conv[7].running_mean,model.conv[7].running_var,model.conv[7].weight,model.conv[7].bias,model.conv[7].eps)
    bn_var_sqrt=torch.sqrt(running2.running_var + running2.eps)
    idbn3.weight.data=bn_var_sqrt
    idbn3.bias.data=running2.running_mean
    idbn3.running_mean.data=running2.running_mean+bias
    idbn3.running_var.data=running2.running_var
    return [idconv1,idbn1,idrelu1,idconv2,idbn2,idrelu2,idconv3,idbn3]

def mobilenetv2_to_mobilenetv1(model):
    features=[]
    for m in model.features:
        if isinstance(m,InvertedResidual)and m.use_res_connect:
                features+=rm_r(m)
        else:
            for mm in m.modules():
                if not list(mm.children()):
                    features.append(mm)

    new_features=[]
    while features:
        if isinstance(features[0],nn.Conv2d) and isinstance(features[1],nn.BatchNorm2d) and isinstance(features[2],nn.Conv2d) and isinstance(features[3],nn.BatchNorm2d):
            conv,bn = fuse_cbcb(features[0],features[1],features[2],features[3])
            new_features.append(conv)
            new_features.append(bn)
            features=features[4:]
        else:
            new_features.append(features.pop(0))
    
    model.features=nn.Sequential(*new_features)
    return model
