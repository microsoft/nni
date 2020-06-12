# Analysis Utils for Model Compression
We provide several easy-to-use tools for users to analyze their model during model compression.

## Sensitivity
First, we provide a sensitivity analysis tool (**SensitivityAnalysis**) for users to analyze the sensitivity of each convolutional layer in their model. Specifically, the SensitiviyAnalysis gradually prune each layer of the model, and test the accuracy of the model at the same time. Note that, SensitivityAnalysis only prunes a layer once a time, and the other layers are set to their original weights. According to the accuracies of different convolutional layers under different sparsities, we can easily find out which layers the model accuracy is more sensitive to. 

### Usage

Following codes show the basic usage of the SensitivityAnalysis.
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

Two key parameters of SensitivityAnalysis are model, and val_func. 'model' is the neural network that to be analyzed and the 'val_func' is the validation function that returns the model accuracy on the validation dataset. Due to different scenarios may have different ways to calculate the loss/accuracy, so users should prepare a function that returns the model accuracy on the dataset and pass it to SensitivityAnalysis.
SensitivityAnalysis can export the sensitivity results as a csv file usage is shown in the example above.

Futhermore, users can specify the sparsities values used to prune for each layer by optinal parameter 'sparsities'.
```python
s_analyzer = SensitivityAnalysis(model=net, val_func=val, sparsities=[0.25, 0.5, 0.75])
``` 
the SensitivityAnalysis will prune 25% 50% 75% weights gradually for each layer, and record the model's accuracy at the same time (SensitivityAnalysis only prune a layer once a time, the other layers are set to their original weights). If the sparsities is not set, SensitivityAnalysis will use the numpy.arange(0.1, 1.0, 0.1) as the default sparsity values.

Users can also speed up the progress of sensitivity analysis by the early_stop option. By default, the SensitivityAnalysis will test the accuracy under all sparsities for each layer. In contrast, when the early_stop is set, the sensitivity analysis for a layer will stop, when the accuracy drop already reaches the threshold set by early_stop.
```python
s_analyzer = SensitivityAnalysis(model=net, val_func=val, sparsities=[0.25, 0.5, 0.75], early_stop=0.1)
```

### Output example
The following lines are the example csv file exported from SensitivityAnalysis. The first line is constructed by 'layername' and sparsity list. Here the sparsity value means how much weight SensitivityAnalysis prune for each layer. Each line below records the model accuracy when this layer is under different sparsities. Note that, due to the early_stop option, some layers may
not have model accuracies under all sparsities, because its accuracy drop has alreay exceeded the threshold set by the user.
```
layername,0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.85,0.95
features.0,0.54566,0.46308,0.06978,0.0374,0.03024,0.01512,0.00866,0.00492,0.00184
features.3,0.54878,0.51184,0.37978,0.19814,0.07178,0.02114,0.00438,0.00442,0.00142
features.6,0.55128,0.53566,0.4887,0.4167,0.31178,0.19152,0.08612,0.01258,0.00236
features.8,0.55696,0.54194,0.48892,0.42986,0.33048,0.2266,0.09566,0.02348,0.0056
features.10,0.55468,0.5394,0.49576,0.4291,0.3591,0.28138,0.14256,0.05446,0.01578
```

## Topology
We also provide several tools for the topology analysis during the model compression.

### ChannelDependency
Complicated models may has residual connection/concat operations in their models. When the user prune these models, they need to be careful about the channel-count dependencies between the convolution layers in the model. If the layers has channel dependency are assigned with different sparsities (here we only discuss the structured pruning by L1FilterPruner/L2FilterPruner), then even the pruned model with mask works fine. but the pruned model cannot be speedup to the final model that run on the devices, because there will be a shape conflict when the model try to add/concat the outputs of these layers. This model is to find the layers that has channel count dependencies to help user better prune their model.

#### Usage
```python
from nni.compression.torch.utils.shape_dependency import ChannelDependency
data = torch.ones(1, 3, 224, 224).cuda()
channel_depen = ChannelDependency(net, data)
channel_depen.export('dependency.csv')
```

#### Output Example
Following lines are the output example of torchvision.models.resnet18 exported by ChannelDependency. The layers at the same line have output channel dependencies with each other. For example, layer1.1.conv2, conv1, and layer1.0.conv2 have output channel dependencies with each other, which means the output channel(filters) numbers of these three layers should be same with each other, otherwise the model may has shape conflict. 
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

### MaskConflict
When the masks of different layers in a model has conflict, we can fix the mask conflict by MaskConflict. Specifically, the MaskConflict loads the masks exported by the pruners(L1FilterPruner, etc), and check if there is mask conflict, if so, MaskConflict sets the conflicting masks to the same value.

```
from nni.compression.torch.utils.mask_conflict import MaskConflict
mc = MaskConflict('./resnet18_mask', net, data)
mc.fix_mask_conflict()
mc.export('./resnet18_fixed_mask')
```