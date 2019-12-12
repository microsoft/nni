# NNI 上的自动模型压缩

使用 NNI 的压缩和 Tuner 能轻松实现自动模型压缩

## 首先，使用 NNI 压缩模型

可使用 NNI 轻松压缩模型。 以剪枝为例，可通过 LevelPruner 对预训练模型剪枝：

```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```

op_type 为 'default' 表示模块类型定义在了 [default_layers.py](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/default_layers.py)。

因此 `{ 'sparsity': 0.8, 'op_types': ['default'] }` 表示 **所有指定 op_types 的层都会被压缩到 0.8 的稀疏度**。 当调用 `pruner.compress()` 时，模型会通过掩码进行压缩。随后还可以微调模型，此时**被剪除的权重不会被更新**。

## 然后，进行自动化

前面的示例手工选择了 LevelPruner，并对所有层使用了相同的稀疏度，显然这不是最佳方法，因为不同层会有不同的冗余度。 每层的稀疏度都应该仔细调整，以便减少模型性能的下降，可通过 NNI Tuner 来完成。

首先需要设计搜索空间，这里使用了嵌套的搜索空间，其中包含了选择的剪枝函数以及需要优化稀疏度的层。

```json
{
  "prune_method": {
    "_type": "choice",
    "_value": [
      {
        "_name": "agp",
        "conv0_sparsity": {
          "_type": "uniform",
          "_value": [
            0.1,
            0.9
          ]
        },
        "conv1_sparsity": {
          "_type": "uniform",
          "_value": [
            0.1,
            0.9
          ]
        },
      },
      {
        "_name": "level",
        "conv0_sparsity": {
          "_type": "uniform",
          "_value": [
            0.1,
            0.9
          ]
        },
        "conv1_sparsity": {
          "_type": "uniform",
          "_value": [
            0.01,
            0.9
          ]
        },
      }
    ]
  }
}
```

然后需要修改几行代码。

```python
import nni
from nni.compression.torch import *
params = nni.get_parameters()
conv0_sparsity = params['prune_method']['conv0_sparsity']
conv1_sparsity = params['prune_method']['conv1_sparsity']
# these raw sparsity should be scaled if you need total sparsity constrained
config_list_level = [{ 'sparsity': conv0_sparsity, 'op_name': 'conv0' },
                     { 'sparsity': conv1_sparsity, 'op_name': 'conv1' }]
config_list_agp = [{'initial_sparsity': 0, 'final_sparsity': conv0_sparsity,
                    'start_epoch': 0, 'end_epoch': 3,
                    'frequency': 1,'op_name': 'conv0' },
                   {'initial_sparsity': 0, 'final_sparsity': conv1_sparsity,
                    'start_epoch': 0, 'end_epoch': 3,
                    'frequency': 1,'op_name': 'conv1' },]
PRUNERS = {'level':LevelPruner(model, config_list_level), 'agp':AGP_Pruner(model, config_list_agp)}
pruner = PRUNERS(params['prune_method']['_name'])
pruner.compress()
... # fine tuning
acc = evaluate(model) # evaluation
nni.report_final_results(acc)
```

最后，定义任务，并使用任务来自动修剪层稀疏度。

```yaml
authorName: default
experimentName: Auto_Compression
trialConcurrency: 2
maxExecDuration: 100h
maxTrialNum: 500
# 可选项: local, remote, pai
trainingServicePlatform: local
# 可选项: true, false
useAnnotation: False
searchSpacePath: search_space.json
tuner:
  # 可选项: TPE, Random, Anneal...
  builtinTunerName: TPE
  classArgs:
    # 可选项: maximize, minimize
    optimize_mode: maximize
trial:
  command: bash run_prune.sh
  codeDir: .
  gpuNum: 1

```

