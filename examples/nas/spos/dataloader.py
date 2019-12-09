import os

import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch.utils.data
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop)
        self.twist = ops.ColorTwist(device="gpu")
        self.jitter_rng = ops.Uniform(range=[0.6, 1.4])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        saturation = self.jitter_rng()
        contrast = self.jitter_rng()
        brightness = self.jitter_rng()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        images = self.twist(images, saturation=saturation, contrast=contrast, brightness=brightness)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(split, image_dir, batch_size, num_threads,
                           crop=224, val_size=256):
    world_size, local_rank = 1, 0
    device_id = torch.cuda.device_count() - 1  # use last gpu
    if split == "train":
        pipeline = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                   data_dir=os.path.join(image_dir, "train"),
                                   crop=crop, world_size=world_size, local_rank=local_rank)
    elif split == "val":
        pipeline = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                 data_dir=os.path.join(image_dir, "val"),
                                 crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
    else:
        raise AssertionError
    pipeline.build()
    num_samples = pipeline.epoch_size("Reader")
    return DALIClassificationIterator(pipeline, size=num_samples, fill_last_batch=split == "train"), \
           (num_samples + batch_size - 1) // batch_size
