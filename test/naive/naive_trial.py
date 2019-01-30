import time

import nni

params = nni.get_parameters()
print('params:', params)
x = params['x']

for i in range(1, 10):
    nni.report_intermediate_result(x ** i)
    time.sleep(0.5)

nni.report_final_result(x ** 10)
