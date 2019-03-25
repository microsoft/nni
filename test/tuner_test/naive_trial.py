import nni

params = nni.get_parameters()
print('params:', params)
x = params['x']

nni.report_final_result(x)
