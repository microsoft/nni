# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import nni
import time

if __name__ == '__main__':
    nni.get_next_parameter()
    time.sleep(3)
    nni.report_final_result(0.5)
