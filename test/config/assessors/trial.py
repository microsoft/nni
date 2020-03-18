# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import time
import nni

if __name__ == '__main__':
    print('trial start')
    if random.random() > 0.5:
        up = True
    else:
        up = False
    v = 0.5
    nni.get_next_parameter()
    for i in range(20):
        time.sleep(1)
        for _ in range(2):
            if up:
                v *= 1.1
            else:
                v *= 0.9
            nni.report_intermediate_result(v)
    nni.report_final_result(v)
    print('trial done')
