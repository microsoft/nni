import nni

params = nni.get_next_parameter()
print('params:', params)
x = params['x']

nni.report_final_result(x)
