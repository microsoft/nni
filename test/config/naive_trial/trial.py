import random
import time
import math
import nni

curve_func = {
    0: lambda x: x,
    1: lambda x: x * x,
    2: lambda x: math.pow(x, 0.5),
    3: lambda x: math.tanh(x)
}

if __name__ == '__main__':
    print('trial start')

    params = nni.get_next_parameter()
    print('params:', params)
    epochs = 20

    for i in range(epochs):
        v = curve_func[params['k']](i / epochs)
        v += v * (random.random() * params['n'])
        v *= params['d']
        nni.report_intermediate_result(v)

        if i % 5 == 0:
            time.sleep(1)
    nni.report_final_result(v)
    print('trial done')
