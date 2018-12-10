import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import curvefunctions

class curvefitting(object):
    def __init__(self, curve_history, best_performance, target_pos = TARGET_POS):
        self.curve_history = curve_history
        self.num = len(curve_history)
        self.best_performance = best_performance
        self.target_pos = target_pos

    def run(self):
        for i in range(NUM_OF_FUNCTIONS):
            init_var = 0x3f3f3f3f
            for a in np.arange (model_para[curve_combination_models[i]][0] - 1.0, model_para[curve_combination_models[i]][0] + 1.0, STEP_SIZE):
                for b in np.arange (model_para[curve_combination_models[i]][1] - 1, model_para[curve_combination_models[i]][1] + 1, STEP_SIZE):
                    for c in np.arange (model_para[curve_combination_models[i]][2] - 1, model_para[curve_combination_models[i]][2] + 1, STEP_SIZE):
                        for d in np.arange (model_para[curve_combination_models[i]][3] - 1, model_para[curve_combination_models[i]][3] + 1, STEP_SIZE):
                            diff = 0
                            for x in range(self.num):
                                y = all_models[curve_combination_models[i]](x + 1, a, b, c, d)
                                diff += (y - self.curve_history[x]) * (y - self.curve_history[x])
                            if diff < init_var:
                                init_var = diff
                                model_fit[curve_combination_models[i]] = [a, b, c, d]
        ans = 0
        for i in range(NUM_OF_FUNCTIONS):
            model = curve_combination_models[i]
            ans += all_models[model](20, model_fit[model][0], model_fit[model][0], model_fit[model][0], model_fit[model][0])

        return ans / 5




if __name__ == "__main__":
    id = 0
    # Test 1
    ret = curvefitting([0.8018166665981213, 0.9276166671514511, 0.94371666431427, 0.9530999948581059], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 2
    ret = curvefitting([0.9412833386659623, 0.9856666787465413, 0.9902166752020518, 0.9932500064373017], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 3
    ret = curvefitting([0.9398166693250338, 0.9840166797240575, 0.9894000099102656, 0.9919666741291682], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 4
    ret = curvefitting([0.0991, 0.3717, 0.5475, 0.6665, 0.7615], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 5
    ret = curvefitting([0.0976, 0.1032, 0.1135, 0.101, 0.1135], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 6
    ret = curvefitting([0.0892, 0.0958, 0.1028, 0.0892, 0.1135, 0.1010], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 7
    ret = curvefitting([0.0958, 0.1032, 0.1032, 0.1032, 0.0974, 0.1009], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 8
    ret = curvefitting([0.1187, 0.9039, 0.9278, 0.9422, 0.9488, 0.9554], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 9
    ret = curvefitting([0.0530, 0.6593, 0.7481, 0.8081, 0.8511, 0.8656], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
    # Test 10
    ret = curvefitting([0.0982, 0.1032, 0.0974, 0.0892, 0.1028, 0.1028], 0.95)
    id += 1
    print ("id = ",id, " Predict y = ", ret.run())
