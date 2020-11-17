# Automatic Model Pruning using NNI Tuners

It's convenient to implement auto model pruning with NNI compression and NNI tuners

## First, model compression with NNI

You can easily compress a model with NNI compression. Take pruning for example, you can prune a pretrained model with LevelPruner like this

```python
from nni.algorithms.compression.pytorch.pruning import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```

The 'default' op_type stands for the module types defined in [default_layers.py](https://github.com/microsoft/nni/blob/v1.9/src/sdk/pynni/nni/compression/pytorch/default_layers.py) for pytorch.

Therefore ```{ 'sparsity': 0.8, 'op_types': ['default'] }```means that **all layers with specified op_types will be compressed with the same 0.8 sparsity**. When ```pruner.compress()``` called, the model is compressed with masks and after that you can normally fine tune this model and **pruned weights won't be updated** which have been masked.

## Then, make this automatic

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
from nni.algorithms.compression.pytorch.pruning import *
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
PRUNERS = {'level':LevelPruner(model, config_list_level), 'agp':AGPPruner(model, config_list_agp)}
pruner = PRUNERS(params['prune_method']['_name'])
pruner.compress()
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
#choice: local, remote, pai
trainingServicePlatform: local
#choice: true, false
useAnnotation: False
searchSpacePath: search_space.json
tuner:
  #choice: TPE, Random, Anneal...
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: bash run_prune.sh
  codeDir: .
  gpuNum: 1

```

