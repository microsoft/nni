# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import numpy as np


# for pruning
def acc_reward(net, acc, flops):
    return acc * 0.01


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)
