# Single Path One-Shot

Single Path One-Shot by Megvii Research.

TODO: Reproduction results.

## Preparation

Need to download the flops lookup table from [here](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN).
Put `op_flops_dict.pkl` and `checkpoint-150000.pth.tar` (if you don't want to retrain the supernet) under `data` directory.

Prepare ImageNet in the standard format (follow the script [here](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4)). Link it to `data/imagenet` will be more convenient.

## Step 1. Train Supernet

```
python supernet.py
```

Will export the checkpoint to checkpoints directory, for the next step.

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

TODO: export final architecture from tuner is not ready yet.

## Step 3. Train from Scratch

```
python scratch.py
```

It will automatically use `architecture_final.json`, which is already included in this repo. You can use any architecture you want with `--fixed-arc` option.
