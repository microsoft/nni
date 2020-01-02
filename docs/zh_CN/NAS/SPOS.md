# 单路径 One-Shot (SPOS)

## 介绍

在 [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) 中提出的 one-shot NAS 方法，通过构造简化的通过统一路径采样方法训练的超网络来解决 One-Shot 模型训练的问题。这样所有架构（及其权重）都得到了完全且平等的训练。 An evolutionary algorithm is then applied to efficiently search for the best-performing architectures without any fine tuning.

Implementation on NNI is based on [official repo](https://github.com/megvii-model/SinglePathOneShot). We implement a trainer that trains the supernet and a evolution tuner that leverages the power of NNI framework that speeds up the evolutionary search phase. We have also shown

## Examples

Here is a use case, which is the search space in paper, and the way to use flops limit to perform uniform sampling.

[Example code](https://github.com/microsoft/nni/tree/master/examples/nas/spos)

### Requirements

NVIDIA DALI >= 0.16 is needed as we use DALI to accelerate the data loading of ImageNet. [Installation guide](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html)

Download the flops lookup table from [here](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN) (maintained by [Megvii](https://github.com/megvii-model)). Put `op_flops_dict.pkl` and `checkpoint-150000.pth.tar` (if you don't want to retrain the supernet) under `data` directory.

Prepare ImageNet in the standard format (follow the script [here](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4)). Linking it to `data/imagenet` will be more convenient.

After preparation, it's expected to have the following code structure:

```
spos
├── architecture_final.json
├── blocks.py
├── config_search.yml
├── data
│   ├── imagenet
│   │   ├── train
│   │   └── val
│   └── op_flops_dict.pkl
├── dataloader.py
├── network.py
├── readme.md
├── scratch.py
├── supernet.py
├── tester.py
├── tuner.py
└── utils.py
```

### Step 1. Train Supernet

```
python supernet.py
```

Will export the checkpoint to `checkpoints` directory, for the next step.

NOTE: The data loading used in the official repo is [slightly different from usual](https://github.com/megvii-model/SinglePathOneShot/issues/5), as they use BGR tensor and keep the values between 0 and 255 intentionally to align with their own DL framework. The option `--spos-preprocessing` will simulate the behavior used originally and enable you to use the checkpoints pretrained.

### Step 2. Evolution Search

Single Path One-Shot leverages evolution algorithm to search for the best architecture. The tester, which is responsible for testing the sampled architecture, recalculates all the batch norm for a subset of training images, and evaluates the architecture on the full validation set.

In order to make the tuner aware of the flops limit and have the ability to calculate the flops, we created a new tuner called `EvolutionWithFlops` in `tuner.py`, inheriting the tuner in SDK.

To have a search space ready for NNI framework, first run

```
nnictl ss_gen -t "python tester.py"
```

This will generate a file called `nni_auto_gen_search_space.json`, which is a serialized representation of your search space.

By default, it will use `checkpoint-150000.pth.tar` downloaded previously. In case you want to use the checkpoint trained by yourself from the last step, specify `--checkpoint` in the command in `config_search.yml`.

Then search with evolution tuner.

```
nnictl create --config config_search.yml
```

The final architecture exported from every epoch of evolution can be found in `checkpoints` under the working directory of your tuner, which, by default, is `$HOME/nni/experiments/your_experiment_id/log`.

### Step 3. Train from Scratch

```
python scratch.py
```

By default, it will use `architecture_final.json`. This architecture is provided by the official repo (converted into NNI format). You can use any architecture (e.g., the architecture found in step 2) with `--fixed-arc` option.

## Reference

### PyTorch

```eval_rst
..  autoclass:: nni.nas.pytorch.spos.SPOSEvolution
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.spos.SPOSSupernetTrainer
    :members:

    .. automethod:: __init__

..  autoclass:: nni.nas.pytorch.spos.SPOSSupernetTrainingMutator
    :members:

    .. automethod:: __init__
```

## Known Limitations

* Block search only. Channel search is not supported yet.
* Only GPU version is provided here.

## Current Reproduction Results

Reproduction is still undergoing. Due to the gap between official release and original paper, we compare our current results with official repo (our run) and paper.

* Evolution phase is almost aligned with official repo. Our evolution algorithm shows a converging trend and reaches ~65% accuracy at the end of search. Nevertheless, this result is not on par with paper. For details, please refer to [this issue](https://github.com/megvii-model/SinglePathOneShot/issues/6).
* Retrain phase is not aligned. Our retraining code, which uses the architecture released by the authors, reaches 72.14% accuracy, still having a gap towards 73.61% by official release and 74.3% reported in original paper.
