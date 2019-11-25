# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import nni

if __name__ == '__main__':
    for i in range(5):
        hyper_params = nni.get_next_parameter()
        print('hyper_params:[{}]'.format(hyper_params))
        if hyper_params is None:
            break
        nni.report_final_result(0.1*i)
        time.sleep(3)
