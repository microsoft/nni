"""
Hosting some commonly used datasets in tests on NNI blob.

If --sas-token is set, the script will attempt to upload archives.
See YAMLs in pipelines/ folder for instructions on how to generate an SAS token.
"""

import argparse
import glob
import os
import random
import shutil
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import CIFAR10, MNIST, ImageNet

IMAGENET_DIR = Path(os.environ.get('IMAGENET_DIR', '/mnt/data/imagenet'))
# Should contain meta.bin, val, and (optionally) train
# val should contain 1000 folders: n01440764, n01443537, ...


def prepare_cifar10(data_dir: Path):
    print('Preparing CIFAR10...')
    CIFAR10(str(data_dir / 'cifar10'), download=True)
    for file in glob.glob(str(data_dir / 'cifar10' / '**' / '*.gz'), recursive=True):
        Path(file).unlink()


def prepare_mnist(data_dir: Path):
    print('Preparing MNIST...')
    MNIST(str(data_dir / 'mnist'), download=True)
    for file in glob.glob(str(data_dir / 'mnist' / '**' / '*.gz'), recursive=True):
        Path(file).unlink()


def prepare_imagenet_subset(data_dir: Path, imagenet_dir: Path):
    print('Preparing ImageNet subset...')
    random_state = random.Random(42)

    # Use a predownloaded imagenet. Only validation set is used.
    imagenet = ImageNet(imagenet_dir, split='val')
    # Randomly select 10% of the data in each category
    images = defaultdict(list)
    for image_path, category_id in imagenet.imgs:
        images[category_id].append(image_path)

    # Target root dir
    subset_dir = data_dir / 'imagenet'
    shutil.rmtree(subset_dir, ignore_errors=True)
    copied_count = 0
    for category_id, imgs in images.items():
        random_state.shuffle(imgs)
        for img in imgs[:len(imgs) // 10]:
            folder_name = Path(img).parent.name
            file_name = Path(img).name
            (subset_dir / folder_name).mkdir(exist_ok=True, parents=True)
            shutil.copyfile(img, subset_dir / folder_name / file_name)
            copied_count += 1
    print(f'Generated a subset of {copied_count} images.')


def zip_datasets(data_dir: Path):
    datasets = [d for d in data_dir.iterdir() if d.is_dir()]
    for dataset in datasets:
        dataset_name = dataset.name
        print(f'Creating archive for {dataset}...')
        shutil.make_archive(str(data_dir / dataset_name), 'zip', data_dir, dataset_name)


def upload_datasets(sas_token):
    if not sas_token:
        warnings.warn('sas_token is not set. Upload is skipped.')
        return
    subprocess.run([
        'azcopy',
        'copy',
        'data/*.zip',
        'https://nni.blob.core.windows.net/testdata/?' + sas_token,
    ], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-generate', default=False, action='store_true')
    parser.add_argument('--sas-token', default=None, type=str)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--imagenet-dir', default='/mnt/data/imagenet', type=str)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.no_generate:
        print('Generation is skipped.')
    else:
        prepare_cifar10(data_dir)
        prepare_mnist(data_dir)
        prepare_imagenet_subset(data_dir, Path(args.imagenet_dir))
        zip_datasets(data_dir)
    upload_datasets(args.sas_token)


if __name__ == '__main__':
    main()
