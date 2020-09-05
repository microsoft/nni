import time
import nni

if __name__ == '__main__':
    print('trial start')
    params = nni.get_next_parameter()
    print('params:', params)
    epochs = 2

    for i in range(epochs):
        nni.report_intermediate_result(0.1 * (i+1))
        time.sleep(1)
    nni.report_final_result(0.8)
    print('trial done')
