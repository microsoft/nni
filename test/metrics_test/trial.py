import time
import nni

if __name__ == '__main__':
    nni.get_next_parameter()
    for i in range(10):
        if i % 2 == 0:
            print('report intermediate result without end of line.', end='')
        else:
            print('report intermediate result.')
        nni.report_intermediate_result(0.1*(i+1))
        time.sleep(2)
    print('report final result')
    if int(nni.get_sequence_id()) % 2 == 0:
        print('test final metrics not at line start.', end='')
    nni.report_final_result(1.0)
    print('done')
