import time
import nni

if __name__ == '__main__':
    for i in range(5):
        hyper_params = nni.get_next_parameter()
        nni.report_final_result(0.1*i)
        time.sleep(3)
