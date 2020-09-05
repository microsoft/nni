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
在此例中，只会分析 `Conv1` 层。 另外，也可以通过并行启动多个进程，将同一个模型的不同层分给每个进程来加速。


### 输出示例
下面是从 SensitivityAnalysis 中导出的 csv 文件示例。 第一行由 'layername' 和稀疏度值的列表组成。 稀疏度值表示 SensitivityAnalysis 为每一层剪枝的权重比例。 每行表示某层在不同稀疏度下的模型精度。 注意，根据 early_stop 选项，某些层可能不会有所有稀疏度下的精度或损失值。比如，精度下降的值超过了定义的阈值。
```
layername,0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.85,0.95
features.0,0.54566,0.46308,0.06978,0.0374,0.03024,0.01512,0.00866,0.00492,0.00184
features.3,0.54878,0.51184,0.37978,0.19814,0.07178,0.02114,0.00438,0.00442,0.00142
features.6,0.55128,0.53566,0.4887,0.4167,0.31178,0.19152,0.08612,0.01258,0.00236
features.8,0.55696,0.54194,0.48892,0.42986,0.33048,0.2266,0.09566,0.02348,0.0056
features.10,0.55468,0.5394,0.49576,0.4291,0.3591,0.28138,0.14256,0.05446,0.01578
```

## 拓扑结构分析
NNI 还提供了在模型压缩过程中，进行模型拓扑分析的工具。 这些工具可帮助用户更好的压缩模型。 压缩模型时，因为网络结构的复杂性，经常需要花时间检查压缩配置是否合理。 因此，NNI 提供了这些工具用于模型拓扑分析，来减轻用户负担。

### ChannelDependency
复杂模型中还会有残差或连接的操作。 对这些模型剪枝时，需要小心卷积层之间通道数量的依赖关系。 以 resnet18 中残差模块为例。 `layer2.0.conv2` 和 `layer2.0.downsample.0` 层输出的特征会加到一起，所以 `layer2.0.conv2` 和 `layer2.0.downsample.0` 的输出通道数量必须一样，否则会有 Tensor 形状的冲突。

![](../../img/channel_dependency_example.jpg)


如果有通道依赖的图层，被分配了不同的稀疏度 (此处仅讨论 L1FilterPruner/L2FilterPruner 的结构化剪枝)，就会造成形状冲突。 即使剪枝后的掩码模型也能正常使用，剪枝后的模型也因为模型在加和、连接这些层的输出时有冲突，不能在设备上加速。 此工具可用于查找有通道依赖的层，帮助更好的剪枝模型。

#### 用法
```python
from nni.compression.torch.utils.shape_dependency import ChannelDependency
data = torch.ones(1, 3, 224, 224).cuda()
channel_depen = ChannelDependency(net, data)
channel_depen.export('dependency.csv')
```

#### Output Example
下列代码是 由 ChannelDependency 导出的 torchvision.models.resnet18 示例。 每行上，有相互依赖的输出通道。 例如，layer1.1.conv2, conv1 和 layer1.0.conv2 相互间有输出依赖。这表示这三个层的输出通道（滤波器）数量需要一致，否则模型会产生形状冲突。
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
当不同层的掩码有冲突时，（例如，为通道依赖的层设置了不同的稀疏度），可通过 MaskConflict 来修复。 即，MaskConflict 可加载由 (L1FilterPruner, 等) 导出的掩码，并检查是否有掩码冲突。如果有 MaskConflict 会将冲突的掩码设置为相同的值。

```
from nni.compression.torch.utils.mask_conflict import fix_mask_conflict
fixed_mask = fix_mask_conflict('./resnet18_mask', net, data)
```

### 模型 FLOPs 和参数计数器
NNI 提供了模型计数器，用于计算模型的 FLOPs 和参数。 此计数器支持计算没有掩码模型的 FLOPs、参数，也可以计算有掩码模型的 FLOPs、参数，这有助于在模型压缩过程中检查模型的复杂度。 注意，对于结构化的剪枝，仅根据掩码来标识保留的滤波器，不会考虑剪枝的输入通道，因此，计算出的 FLOPs 会比实际数值要大（即，模型加速后的计算值）。

### 用法
```
from nni.compression.torch.utils.counter import count_flops_params

# Given input size (1, 1, 28, 28) 
flops, params = count_flops_params(model, (1, 1, 28, 28))
# Format output size to M (i.e., 10^6)
print(f'FLOPs: {flops/1e6:.3f}M,  Params: {params/1e6:.3f}M)
```