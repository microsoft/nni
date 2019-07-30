import os
import time

import nni

params = nni.get_next_parameter()
print('params:', params)
x = params['x']

if x == 2:
    lock_path = os.path.join(os.path.dirname(__file__), 'resubmit.lock')
    try:
        open(lock_path, 'x')
        exit(1)
    except FileExistsError:
        pass

for i in range(1, 10):
    nni.report_intermediate_result(x ** i)
    time.sleep(0.5)

nni.report_final_result(x ** 10)
