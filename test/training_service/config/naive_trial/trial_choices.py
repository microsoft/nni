import random
import nni

if __name__ == '__main__':
    print('trial start')

    params = nni.get_next_parameter()
    print('params:', params)

    nni.report_intermediate_result(random.random())
    nni.report_final_result(random.random())

    print('trial done')
