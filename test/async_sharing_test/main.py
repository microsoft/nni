# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Test code for weight sharing
need NFS setup and mounted as `/mnt/nfs/nni`
"""

import hashlib
import os
import random
import time

import nni


def generate_rand_file(fl_name):
    """
    generate random file and write to `fl_name`
    """
    fl_size = random.randint(1024, 102400)
    fl_dir = os.path.split(fl_name)[0]
    if not os.path.exists(fl_dir):
        os.makedirs(fl_dir)
    with open(fl_name, 'wb') as fout:
        fout.write(os.urandom(fl_size))


def check_sum(fl_name, tid=None):
    """
    compute checksum for generated file of `fl_name`
    """
    hasher = hashlib.md5()
    with open(fl_name, 'rb') as fin:
        for chunk in iter(lambda: fin.read(4096), b""):
            hasher.update(chunk)
    ret = hasher.hexdigest()
    if tid is not None:
        ret = ret + str(tid)
    return ret


if __name__ == '__main__':
    nfs_path = '/mnt/nfs/nni/test'
    params = nni.get_next_parameter()
    print(params)
    if params['id'] == 0:
        model_file = os.path.join(nfs_path, str(params['id']), 'model.dat')
        generate_rand_file(model_file)
        time.sleep(10)
        nni.report_final_result({
            'checksum': check_sum(model_file, tid=params['id']),
            'path': model_file
        })
    else:
        model_file = params['prev_path']
        time.sleep(10)
        nni.report_final_result({
            'checksum': check_sum(model_file, tid=params['prev_id'])
        })
