# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

METRICS_FILENAME = '.nni/metrics'
MAGIC = 'ME'

def sdk_send_data(data):
    out_dir = os.getenv('NNI_SYS_DIR')
    if not os.path.isdir(out_dir):
        raise Exception('Can not find NNI_SYS_DIR: {}'.format(out_dir))

    filename = os.path.join(out_dir, METRICS_FILENAME)
    wrapped_data = data + '\n'
    datalen = len(wrapped_data)
    if datalen < 2:
        return
    with open(filename, 'a') as f:
        f.write('ME{:06d}{}'.format(datalen, wrapped_data))

def user_code():

    epochs = 20

    val_acc = 0
    batch_size = 32
    for epoch in range(epochs):
        #Training
        time.sleep(1)
        val_acc += 0.5
        metrics = 'epoch: {}, val accuracy: {:.2f}, batch size: {}'.format(epoch, val_acc, batch_size)
        sdk_send_data(metrics)

if __name__ == '__main__':
    print('>>>start...')
    user_code()
    print('>>>end...')
