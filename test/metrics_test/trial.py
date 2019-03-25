import time
import nni

if __name__ == '__main__':
    hyper_params = nni.get_next_parameter()

    for i in range(10):    
        nni.report_intermediate_result(0.1*(i+1))
        time.sleep(2)
    nni.report_final_result(1.0)
