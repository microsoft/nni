import time
from utils import *
def measure_time(model, dummy_input, runtimes=200):
    times = []
    with torch.no_grad():
        for runtime in range(runtimes):
            torch.cuda.synchronize()
            start = time.time()
            out=model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean*1000, std*1000

model = create_model('mobilenet_v1').cpu()
data = torch.rand(32,3,224,224).cpu()
trace = torch.jit.trace(model, data)
print(measure_time(trace,data))

import pdb; pdb.set_trace()
torch.cuda.mem_get_info()

