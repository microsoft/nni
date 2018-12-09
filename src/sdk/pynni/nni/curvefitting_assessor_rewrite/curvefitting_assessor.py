import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import curvefunctions

# constant
NUM_OF_FUNCTIONS = 10
# variable
NUM_OF_SIMULATION_TIME = 5 #20000
NUM_OF_INSTANCE = 2 #500
STEP_SIZE = 2
TARGET_POS = 20
THRESHOLD = 0.05

class mcmc_sampling(object):
    def __init__(self, curve_history, best_performance = 0, target_pos = TARGET_POS):
        # TODO(Shufan): make sure number of curve_history > 0
        self.curve_history = curve_history
        self.history_num = len(curve_history)
        print ("history_lenth = ", self.history_num)
        init_xi = np.ones((NUM_OF_FUNCTIONS), dtype=np.float) / NUM_OF_FUNCTIONS
        for model in curve_combination_models:
            init_xi = np.concatenate((init_xi, model_para[model]))
        self.xi_dim = len(init_xi)
        self.samples = np.broadcast_to(init_xi, (NUM_OF_INSTANCE, self.xi_dim))
        print (self.samples)
        self.target_pos = target_pos
        self.best_performance = best_performance
    
    def f_comb(self, x, Xi):
        # TODO(Shufan)：calculation f_comb once a time (when update Xi)
        print ("x = ", x)
        ret = np.zeros(NUM_OF_INSTANCE)
        for i in range (NUM_OF_INSTANCE):
            idx = NUM_OF_FUNCTIONS
            for j in range (NUM_OF_FUNCTIONS):
                if dimention_of_para[j] == 2:
                    ret += Xi[i][j] * all_models[curve_combination_models[j]](x, Xi[i][idx], Xi[i][idx + 1])
                    idx += 2
                elif dimention_of_para[j] == 3:
                    print ("Xi = ", Xi)
                    ret += Xi[i][j] * all_models[curve_combination_models[j]](x, Xi[i][idx], Xi[i][idx + 1], Xi[i][idx + 2])
                    idx += 3
                elif dimention_of_para[j] == 4:
                    ret += Xi[i][j] * all_models[curve_combination_models[j]](x, Xi[i][idx], Xi[i][idx + 1], Xi[i][idx + 2], Xi[i][idx + 3])
                    idx += 4
                print ("idx = ", idx, "value = ", Xi[i][idx])
        return ret

    def sigma_sq(self, Xi):
        # TODO(Shufan): calculate sigma once a time
        ret = 0
        for i in range(self.history_num):
            temp = self.curve_history[i] - self.f_comb(i, Xi)
            ret += temp * temp
        return 1.0 * ret / self.history_num

    def prior(self, Xi):
        if not self.f_comb(1, Xi) < self.f_comb(self.target_pos, Xi):
            return 0
        for i in range(10):
            if not self.samples[i] > 0:
                return 0
        return 1.0 / np.sqrt(self.sigma_sq(Xi))

    def gaussian_distribution(self, x, Xi):
        return np.exp(np.square(self.curve_history[x] - self.f_comb(x, Xi)) / (-2.0 * self.sigma_sq(Xi))) / np.sqrt(2 * np.pi * np.sqrt(self.sigma_sq(Xi)))

    def likelihood(self, Xi):
        ret = 1
        for i in range(self.history_num):
            ret *= self.gaussian_distribution(i, Xi)
        return ret

    def target_distribution(self, Xi):
        return self.likelihood(Xi) * self.prior(Xi)

    def HM_sampling(self):
        for i in range(NUM_OF_SIMULATION_TIME):
            # get new Instance
            new_values = np.random.randn(NUM_OF_INSTANCE, self.xi_dim) * STEP_SIZE + self.samples

            alpha = np.minimum(1, self.target_distribution(new_values) / self.target_distribution(self.samples))

            u = np.random.rand(NUM_OF_INSTANCE, self.xi_dim)

            change_value_flag = (u < alpha).astype(np.int)

            self.samples = self.samples * (1 - change_value_flag) + new_values * change_value_flag

            # print (self.samples)

    def generate_expect_y():
        return np.sum(self.f_comb(TARGET_POS)) / NUM_OF_INSTANCE

    def assess():
        greater_num = 0
        y = self.y_comb(TARGET_POS)
        for i in range(NUM_OF_INSTANCE):
            if y[i] > self.best_performance:
                greater_num += 1
        if greater_num / NUM_OF_INSTANCE < THRESHOLD:
            return False
        else:
            return True

if __name__ == "__main__":
    ret = mcmc_sampling([1,2,3,4,5], 0.95)
    ret.HM_sampling()
    print ("Predict y = ", ret.generate_expect_y())
