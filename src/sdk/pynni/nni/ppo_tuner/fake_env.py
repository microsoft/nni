from gym import spaces
import random
import numpy as np

class fake_env(object):
    def __init__(self):
        self.num_envs = 12
        self.observation_space = spaces.Discrete(6)
        self.action_space = spaces.Discrete(6)
        self.cur_step = 0
        self.steps = 128

    def reset(self):
        self.cur_step = 0
        print('fake env reset...')
        return [random.randint(1, 6) for i in range(self.num_envs)]

    def step(self, actions):
        self.cur_step += 1
        #print('---------------zql: steps ', self.cur_step, self.steps)
        if self.cur_step == self.steps:
            done = True
            self.cur_step = 0
        else:
            done = False

        rewards = []
        dones = []
        for each in actions:
            rewards.append(0.8)
            dones.append(done)
        dones = np.asarray(dones, dtype=np.bool_)
        return actions, rewards, dones, None
