# Single Path One-Shot Neural Architecture Search with Uniform Sampling

Single Path One-Shot by Megvii Research. [Paper link](https://arxiv.org/abs/1904.00420) [Official repo](https://github.com/megvii-model/SinglePathOneShot)

Block search only. Channel search is not supported yet.

TODO: Reproduction results.

## Preparation

Need to download the flops lookup table from [here](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN).
Put `op_flops_dict.pkl` and `checkpoint-150000.pth.tar` (if you don't want to retrain the supernet) under `data` directory.

Prepare ImageNet in the standard format (follow the script [here](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4)). Link it to `data/imagenet` will be more convenient.

We don't support SPOS on CPU. You need to have at least one GPU to run the experiment. This is mainly because NVIDIA DALI is used as a prerequisite to accelerate the data loading of ImageNet.

## Step 1. Train Supernet

```
python supernet.py
```

Will export the checkpoint to checkpoints directory, for the next step.

NOTE: The data loading used in the official repo is [slightly different from usual](https://github.com/megvii-model/SinglePathOneShot/issues/5). The option `--spos-preprocessing` will simulate the behavior used originally and enable you to use the checkpoints pretrained.

## Step 2. Evolution Search

To have a search space ready for NNI framework, first run

```
nnictl ss_gen -t "python tester.py"
```

This will generate a file called `nni_auto_gen_search_space.json`, which is a serialized representation of your search space.

Then search with evolution tuner.

```
nnictl create --config config_search.yml
```

The final architecture exported from every epoch of evolution can be found in `checkpoints` under the working directory of your tuner, which, by default, is `$HOME/nni/experiments/$EXP_ID/log`.

## Step 3. Train from Scratch

```
python scratch.py
```

It will automatically use `architecture_final.json`, which is already tracked here. You can use any architecture (e.g., the architecture found in step 2) with `--fixed-arc` option.
