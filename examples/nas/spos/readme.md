# Single Path One-Shot Neural Architecture Search with Uniform Sampling

Single Path One-Shot by Megvii Research. [Paper link](https://arxiv.org/abs/1904.00420). [Official repo](https://github.com/megvii-model/SinglePathOneShot).

Block search only. Channel search is not supported yet.

Only GPU version is provided here.

TODO: Reproduction results.

## Preparation

### Requirements

* PyTorch >= 1.2
* NVIDIA DALI >= 0.16 as we use DALI to accelerate the data loading of ImageNet.

### Data

Need to download the flops lookup table from [here](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN).
Put `op_flops_dict.pkl` and `checkpoint-150000.pth.tar` (if you don't want to retrain the supernet) under `data` directory.

Prepare ImageNet in the standard format (follow the script [here](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4)). Link it to `data/imagenet` will be more convenient.

After preparation, it's expected to have the following code structure:

```
spos
├── architecture_final.json
├── blocks.py
├── config_search.yml
├── data
│   ├── imagenet
│   │   ├── train
│   │   └── val
│   └── op_flops_dict.pkl
├── dataloader.py
├── network.py
├── nni_auto_gen_search_space.json
├── readme.md
├── scratch.py
├── supernet.py
├── tester.py
├── tuner.py
└── utils.py
```

## Step 1. Train Supernet

```
python supernet.py
```

Will export the checkpoint to checkpoints directory, for the next step.

NOTE: The data loading used in the official repo is [slightly different from usual](https://github.com/megvii-model/SinglePathOneShot/issues/5), as they use BGR tensor and keep the values between 0 and 255 intentionally to align with their own DL framework. The option `--spos-preprocessing` will simulate the behavior used originally and enable you to use the checkpoints pretrained.

## Step 2. Evolution Search

Single Path One-Shot leverages evolution algorithm to search for the best architecture. The tester, which is responsible for testing the sampled architecture, recalculates all the batch norm for a subset of training images, and evaluates the architecture on the full validation set.

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

By default, it will use `architecture_final.json`. This architecture is provided by the official repo (converted into NNI format). You can use any architecture (e.g., the architecture found in step 2) with `--fixed-arc` option.
