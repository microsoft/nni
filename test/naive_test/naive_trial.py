# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

import nni

params = nni.get_next_parameter()
print('params:', params)
x = params['x']

for i in range(1, 10):
    print(x, i)
    nni.report_intermediate_result(x ** i)
    time.sleep(0.5)

print('final')
nni.report_final_result(x ** 10)
