# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('NAS benchmark downloader')
    parser.add_argument('benchmark_name', choices=['nasbench101', 'nasbench201', 'nds'])

    args = parser.parse_args()

    from .utils import download_benchmark
    download_benchmark(args.benchmark_name)
