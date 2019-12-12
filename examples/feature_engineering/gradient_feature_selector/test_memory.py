import os

LIBSVM_DATA = {
    "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
    "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
    "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
    "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
    "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
    "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2",
}

pipeline_name = "Tree"
device = "CUDA_VISIBLE_DEVICES=0 "
script = "setsid python -m memory_profiler benchmark_test.py "
test_object = "memory"

for name in LIBSVM_DATA:
    log_name = "_".join([pipeline_name, name, test_object])
    command = device + script + "--pipeline_name " + pipeline_name + " --name " + name + " --object " + test_object + " >" +log_name + " 2>&1 &"
    print("command is\t", command)
    os.system(command)
    print("log is here\t", log_name)

print("Done.")


