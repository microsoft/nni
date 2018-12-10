import hashlib
import os
import random
import time

import nni


def generate_rand_file(fl_name):
    fl_size = random.randint(1024, 102400)
    with open(fl_name, 'wb') as fout:
        fout.write(os.urandom(fl_size))


def check_sum(fl_name, id=None):
    hasher = hashlib.md5()
    with open(fl_name, 'rb') as fin:
        for chunk in iter(lambda: fin.read(4096), b""):
            hasher.update(chunk)
    ret = hasher.hexdigest()
    if id is not None:
        ret = ret + str(id)
    return ret


if __name__ == '__main__':
    nfs_path = '/mount/nfs/shared'
    params = nni.get_parameters()
    if params['prev_id'] == 0:
        model_file = os.path.join(nfs_path, str(params['id'], 'model.dat'))
        time.sleep(10)
        generate_rand_file(model_file)
        nni.report_final_result({
            'checksum': check_sum(model_file),
            'path': model_file
        })
    else:
        model_file = params['prev_path']
        nni.report_final_result({
            'checksum': check_sum(model_file, params['prev_id'])
        })