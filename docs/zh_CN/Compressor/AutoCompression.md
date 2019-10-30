# NNI 上的自动模型压缩

使用 NNI 的压缩和 Tuner 能轻松实现自动模型压缩

## 首先，使用 NNI 压缩模型

可使用 NNI 轻松压缩模型。 以剪枝为例，可通过 LevelPruner 对预训练模型剪枝：

```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(config_list)
pruner(model)
```

The 'default' op_type stands for the module types defined in [default_layers.py](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/compression/torch/default_layers.py) for pytorch.

Therefore `{ 'sparsity': 0.8, 'op_types': ['default'] }`means that **all layers with specified op_types will be compressed with the same 0.8 sparsity**. When `pruner(model)` called, the model is compressed with masks and after that you can normally fine tune this model and **pruned weights won't be updated** which have been masked.

## 然后，进行自动化

The previous example manually choosed LevelPruner and pruned all layers with the same sparsity, this is obviously sub-optimal because different layers may have different redundancy. Layer sparsity should be carefully tuned to achieve least model performance degradation and this can be done with NNI tuners.

The first thing we need to do is to design a search space, here we use a nested search space which contains  choosing pruning algorithm and optimizing layer sparsity.

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

Then we need to modify our codes for few lines

```python
import nni
from nni.compression.torch import *
params = nni.get_parameters()
conv0_sparsity = params['prune_method']['conv0_sparsity']
conv1_sparsity = params['prune_method']['conv1_sparsity']
# 如果对总稀疏度有要求，这些原始稀疏度就需要调整。
config_list_level = [{ 'sparsity': conv0_sparsity, 'op_name': 'conv0' },
                     { 'sparsity': conv1_sparsity, 'op_name': 'conv1' }]
config_list_agp = [{'initial_sparsity': 0, 'final_sparsity': conv0_sparsity,
                    'start_epoch': 0, 'end_epoch': 3,
                    'frequency': 1,'op_name': 'conv0' },
                   {'initial_sparsity': 0, 'final_sparsity': conv1_sparsity,
                    'start_epoch': 0, 'end_epoch': 3,
                    'frequency': 1,'op_name': 'conv1' },]
PRUNERS = {'level':LevelPruner(config_list_level)，'agp':AGP_Pruner(config_list_agp)}
pruner = PRUNERS(params['prune_method']['_name'])
pruner(model)
... # fine tuning
acc = evaluate(model) # evaluation
nni.report_final_results(acc)
```

Last, define our task and automatically tuning pruning methods with layers sparsity

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

