# 模型压缩分析工具

```eval_rst
.. contents::
```

NNI 提供了几种易于使用的工具，在压缩时用于分析模型。

## 灵敏度分析
首先提供的是灵敏度分析工具 (**SensitivityAnalysis**)，用于分析模型中每个卷积层的灵敏度。 具体来说，SensitiviyAnalysis 会为每层逐渐剪枝，同时测试模型的精度变化。 注意，敏感度分析一次只会对一层进行剪枝，其它层会使用它们原始的权重。 根据不同稀疏度下不同卷积层的精度，可以很容易的找出模型精度对哪些层的变化更敏感。

### 用法

下列代码是 SensitivityAnalysis 的基本用法。
```python
from nni.compression.torch.utils.sensitivity_analysis import SensitivityAnalysis

def val(model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batchid, (data, label) in enumerate(val_loader):
            data, label = data.cuda(), label.cuda()
            out = model(data)
            _, predicted = out.max(1)
            total += data.size(0)
            correct += predicted.eq(label).sum().item()
    return correct / total

s_analyzer = SensitivityAnalysis(model=net, val_func=val)
sensitivity = s_analyzer.analysis(val_args=[net])
os.makedir(outdir)
s_analyzer.export(os.path.join(outdir, filename))
```

SensitivityAnalysis 的两个重要参数是 `model`, 和 `val_func`。 `model` 是要分析的神经网络，`val_func` 是返回验证数据集的精度、损失或其它指标的验证函数。 根据不同的场景，可能需要不同的方法来计算损失和精度，因此用户需要定义能返回模型精度、损失的函数，并传给 SensitivityAnalysis。 上面的示例也展示了如何用 SensitivityAnalysis 将敏感度结果导出为 csv 文件。

除此之外，还可以使用可选参数 `sparsities` 来为每一层设置稀疏度值。
```python
s_analyzer = SensitivityAnalysis(model=net, val_func=val, sparsities=[0.25, 0.5, 0.75])
```
SensitivityAnalysis 会为每一层逐渐剪枝 25% 50% 75% 的权重，并同时记录模型精度 (SensitivityAnalysis 一次只修建一层，其他层会使用原始权重)。 如果没有设置稀疏度，SensitivityAnalysis 会将 numpy.arange(0.1, 1.0, 0.1) 作为默认的稀疏度值。

还可以通过 early_stop_mode 和 early_stop_value 选项来加快灵敏度分析。 默认情况下，SensitivityAnalysis 会为每一层测试所有的稀疏度值下的精度。 而设置了 early_stop_mode 和 early_stop_value 后，当精度或损失值到了 early_stop_value 所设置的阈值时，会停止灵敏度分析。 支持的提前终止模式包括：minimize, maximize, dropped, raised。

minimize: 当 val_func 的返回值低于 `early_stop_value` 时，会停止分析。

maximize: 当 val_func 的返回值大于 `early_stop_value` 时，会停止分析。

dropped: 当验证指标下降 `early_stop_value` 时，会停止分析。

raised: 当验证指标增加 `early_stop_value` 时，会停止分析。

```python
s_analyzer = SensitivityAnalysis(model=net, val_func=val, sparsities=[0.25, 0.5, 0.75], early_stop_mode='dropped', early_stop_value=0.1)
```
如果只想分析部分卷积层，可在分析函数中通过 `specified_layers` 指定。 `specified_layers` 是卷积层的 Pytorch 模块名称。 例如：
```python
sensitivity = s_analyzer.analysis(val_args=[net], specified_layers=['Conv1'])
```
在此例中，只会分析 `Conv1` 层。 In addtion, users can quickly and easily achieve the analysis parallelization by launching multiple processes and assigning different conv layers of the same model to each process.


### 输出示例
The following lines are the example csv file exported from SensitivityAnalysis. The first line is constructed by 'layername' and sparsity list. Here the sparsity value means how much weight SensitivityAnalysis prune for each layer. Each line below records the model accuracy when this layer is under different sparsities. Note that, due to the early_stop option, some layers may not have model accuracies/losses under all sparsities, for example, its accuracy drop has already exceeded the threshold set by the user.
```
layername,0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.85,0.95
features.0,0.54566,0.46308,0.06978,0.0374,0.03024,0.01512,0.00866,0.00492,0.00184
features.3,0.54878,0.51184,0.37978,0.19814,0.07178,0.02114,0.00438,0.00442,0.00142
features.6,0.55128,0.53566,0.4887,0.4167,0.31178,0.19152,0.08612,0.01258,0.00236
features.8,0.55696,0.54194,0.48892,0.42986,0.33048,0.2266,0.09566,0.02348,0.0056
features.10,0.55468,0.5394,0.49576,0.4291,0.3591,0.28138,0.14256,0.05446,0.01578
```

## 拓扑结构分析
We also provide several tools for the topology analysis during the model compression. These tools are to help users compress their model better. Because of the complex topology of the network, when compressing the model, users often need to spend a lot of effort to check whether the compression configuration is reasonable. So we provide these tools for topology analysis to reduce the burden on users.

### ChannelDependency
Complicated models may have residual connection/concat operations in their models. When the user prunes these models, they need to be careful about the channel-count dependencies between the convolution layers in the model. Taking the following residual block in the resnet18 as an example. The output features of the `layer2.0.conv2` and `layer2.0.downsample.0` are added together, so the number of the output channels of `layer2.0.conv2` and `layer2.0.downsample.0` should be the same, or there may be a tensor shape conflict.

![](../../img/channel_dependency_example.jpg)


If the layers have channel dependency are assigned with different sparsities (here we only discuss the structured pruning by L1FilterPruner/L2FilterPruner), then there will be a shape conflict during these layers. Even the pruned model with mask works fine, the pruned model cannot be speedup to the final model directly that runs on the devices, because there will be a shape conflict when the model tries to add/concat the outputs of these layers. This tool is to find the layers that have channel count dependencies to help users better prune their model.

#### 用法
```python
from nni.compression.torch.utils.shape_dependency import ChannelDependency
data = torch.ones(1, 3, 224, 224).cuda()
channel_depen = ChannelDependency(net, data)
channel_depen.export('dependency.csv')
```

#### Output Example
The following lines are the output example of torchvision.models.resnet18 exported by ChannelDependency. The layers at the same line have output channel dependencies with each other. For example, layer1.1.conv2, conv1, and layer1.0.conv2 have output channel dependencies with each other, which means the output channel(filters) numbers of these three layers should be same with each other, otherwise, the model may have shape conflict.
```
Dependency Set,Convolutional Layers
Set 1,layer1.1.conv2,layer1.0.conv2,conv1
Set 2,layer1.0.conv1
Set 3,layer1.1.conv1
Set 4,layer2.0.conv1
Set 5,layer2.1.conv2,layer2.0.conv2,layer2.0.downsample.0
Set 6,layer2.1.conv1
Set 7,layer3.0.conv1
Set 8,layer3.0.downsample.0,layer3.1.conv2,layer3.0.conv2
Set 9,layer3.1.conv1
Set 10,layer4.0.conv1
Set 11,layer4.0.downsample.0,layer4.1.conv2,layer4.0.conv2
Set 12,layer4.1.conv1
```

### 掩码冲突
When the masks of different layers in a model have conflict (for example, assigning different sparsities for the layers that have channel dependency), we can fix the mask conflict by MaskConflict. Specifically, the MaskConflict loads the masks exported by the pruners(L1FilterPruner, etc), and check if there is mask conflict, if so, MaskConflict sets the conflicting masks to the same value.

```
from nni.compression.torch.utils.mask_conflict import MaskConflict
mc = MaskConflict('./resnet18_mask', net, data)
mc.fix_mask_conflict()
mc.export('./resnet18_fixed_mask')
```