# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch.utils.data
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, seed=12, local_rank=0, world_size=1,
                 spos_pre=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=seed + device_id)
        color_space_type = types.BGR if spos_pre else types.RGB
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=color_space_type)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop,
                                         interp_type=types.INTERP_LINEAR if spos_pre else types.INTERP_TRIANGULAR)
        self.twist = ops.ColorTwist(device="gpu")
        self.jitter_rng = ops.Uniform(range=[0.6, 1.4])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=color_space_type,
                                            mean=0. if spos_pre else [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=1. if spos_pre else [0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.twist(images, saturation=self.jitter_rng(),
                            contrast=self.jitter_rng(), brightness=self.jitter_rng())
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, seed=12, local_rank=0, world_size=1,
                 spos_pre=False, shuffle=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=seed + device_id)
        color_space_type = types.BGR if spos_pre else types.RGB
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=shuffle)
        self.decode = ops.ImageDecoder(device="mixed", output_type=color_space_type)
        self.res = ops.Resize(device="gpu", resize_shorter=size,
                              interp_type=types.INTERP_LINEAR if spos_pre else types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=color_space_type,
                                            mean=0. if spos_pre else [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=1. if spos_pre else [0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class ClassificationWrapper:
    def __init__(self, loader, size):
        self.loader = loader
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.loader)
        return data[0]["data"], data[0]["label"].view(-1).long().cuda(non_blocking=True)

    def __len__(self):
        return self.size


def get_imagenet_iter_dali(split, image_dir, batch_size, num_threads, crop=224, val_size=256,
                           spos_preprocessing=False, seed=12, shuffle=False, device_id=None):
    world_size, local_rank = 1, 0
    if device_id is None:
        device_id = torch.cuda.device_count() - 1  # use last gpu
    if split == "train":
        pipeline = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                   data_dir=os.path.join(image_dir, "train"), seed=seed,
                                   crop=crop, world_size=world_size, local_rank=local_rank,
                                   spos_pre=spos_preprocessing)
    elif split == "val":
        pipeline = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                 data_dir=os.path.join(image_dir, "val"), seed=seed,
                                 crop=crop, size=val_size, world_size=world_size, local_rank=local_rank,
                                 spos_pre=spos_preprocessing, shuffle=shuffle)
    else:
        raise AssertionError
    pipeline.build()
    num_samples = pipeline.epoch_size("Reader")
    return ClassificationWrapper(
        DALIClassificationIterator(pipeline, size=num_samples, fill_last_batch=split == "train",
                                   auto_reset=True), (num_samples + batch_size - 1) // batch_size)
