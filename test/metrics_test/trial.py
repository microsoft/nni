import time
import nni

if __name__ == '__main__':
    nni.get_next_parameter()
    for i in range(10):
        nni.report_intermediate_result(0.1*(i+1))
        time.sleep(2)
    if int(nni.get_sequence_id()) % 2 == 0:
        print('test metrics not at line start.', end='')
    nni.report_final_result(1.0)
