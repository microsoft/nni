# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
from copy import deepcopy
from argparse import Namespace
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from nni.compression.torch.compressor import Pruner
from .channel_pruning_env import ChannelPruningEnv
from .lib.agent import DDPG
from .lib.utils import get_output_folder

torch.backends.cudnn.deterministic = True

_logger = logging.getLogger(__name__)

class AMCPruner(Pruner):
    """
    A pytorch implementation of AMC: AutoML for Model Compression and Acceleration on Mobile Devices.
    (https://arxiv.org/pdf/1802.03494.pdf)

    Parameters:
        model: nn.Module
            The model to be pruned.
        config_list: list
            Configuration list to configure layer pruning.
            Supported keys:
            - op_types: operation type to be pruned
            - op_names: operation name to be pruned
        evaluator: function
            function to evaluate the pruned model.
            The prototype of the function:
            >>> def evaluator(val_loader, model):
            >>>     ...
            >>>     return acc
        val_loader: torch.utils.data.DataLoader
            Data loader of validation dataset.
        suffix: str
            suffix to help you remember what experiment you ran. Default: None.

        # parameters for pruning environment
        model_type: str
            model type to prune, currently 'mobilenet' and 'mobilenetv2' are supported. Default: mobilenet
        flops_ratio: float
            preserve flops ratio. Default: 0.5
        lbound: float
            minimum weight preserve ratio for each layer. Default: 0.2
        rbound: float
            maximum weight preserve ratio for each layer. Default: 1.0
        reward: function
            reward function type:
            - acc_reward: accuracy * 0.01
            - acc_flops_reward: - (100 - accuracy) * 0.01 * np.log(flops)
            Default: acc_reward
        # parameters for channel pruning
        n_calibration_batches: int
            number of batches to extract layer information. Default: 60
        n_points_per_layer: int
            number of feature points per layer. Default: 10
        channel_round: int
            round channel to multiple of channel_round. Default: 8

        # parameters for ddpg agent
        hidden1: int
            hidden num of first fully connect layer. Default: 300
        hidden2: int
            hidden num of second fully connect layer. Default: 300
        lr_c: float
            learning rate for critic. Default: 1e-3
        lr_a: float
            learning rate for actor. Default: 1e-4
        warmup: int
            number of episodes without training but only filling the replay memory. During warmup episodes,
            random actions ares used for pruning. Default: 100
        discount: float
            next Q value discount for deep Q value target. Default: 0.99
        bsize: int
            minibatch size for training DDPG agent. Default: 64
        rmsize: int
            memory size for each layer. Default: 100
        window_length: int
            replay buffer window length. Default: 1
        tau: float
            moving average for target network being used by soft_update. Default: 0.99
        # noise
        init_delta: float
            initial variance of truncated normal distribution
        delta_decay: float
            delta decay during exploration

        # parameters for training ddpg agent
        max_episode_length: int
            maximum episode length
        output_dir: str
            output directory to save log files and model files. Default: ./logs
        debug: boolean
            debug mode
        train_episode: int
            train iters each timestep. Default: 800
        epsilon: int
            linear decay of exploration policy. Default: 50000
        seed: int
            random seed to set for reproduce experiment. Default: None
    """

    def __init__(
            self,
            model,
            config_list,
            evaluator,
            val_loader,
            suffix=None,
            model_type='mobilenet',
            dataset='cifar10',
            flops_ratio=0.5,
            lbound=0.2,
            rbound=1.,
            reward='acc_reward',
            n_calibration_batches=60,
            n_points_per_layer=10,
            channel_round=8,
            hidden1=300,
            hidden2=300,
            lr_c=1e-3,
            lr_a=1e-4,
            warmup=100,
            discount=1.,
            bsize=64,
            rmsize=100,
            window_length=1,
            tau=0.01,
            init_delta=0.5,
            delta_decay=0.99,
            max_episode_length=1e9,
            output_dir='./logs',
            debug=False,
            train_episode=800,
            epsilon=50000,
            seed=None):

        self.val_loader = val_loader
        self.evaluator = evaluator

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        checkpoint = deepcopy(model.state_dict())

        super().__init__(model, config_list, optimizer=None)

        # build folder and logs
        base_folder_name = '{}_{}_r{}_search'.format(model_type, dataset, flops_ratio)
        if suffix is not None:
            self.output_dir = os.path.join(output_dir, base_folder_name + '-' + suffix)
        else:
            self.output_dir = get_output_folder(output_dir, base_folder_name)

        self.env_args = Namespace(
            model_type=model_type,
            preserve_ratio=flops_ratio,
            lbound=lbound,
            rbound=rbound,
            reward=reward,
            n_calibration_batches=n_calibration_batches,
            n_points_per_layer=n_points_per_layer,
            channel_round=channel_round,
            output=self.output_dir
        )
        self.env = ChannelPruningEnv(
            self, evaluator, val_loader, checkpoint, args=self.env_args)
        _logger.info('=> Saving logs to %s', self.output_dir)
        self.tfwriter = SummaryWriter(log_dir=self.output_dir)
        self.text_writer = open(os.path.join(self.output_dir, 'log.txt'), 'w')
        _logger.info('=> Output path: %s...', self.output_dir)

        nb_states = self.env.layer_embedding.shape[1]
        nb_actions = 1  # just 1 action here

        rmsize = rmsize * len(self.env.prunable_idx)  # for each layer
        _logger.info('** Actual replay buffer size: %d', rmsize)

        self.ddpg_args = Namespace(
            hidden1=hidden1,
            hidden2=hidden2,
            lr_c=lr_c,
            lr_a=lr_a,
            warmup=warmup,
            discount=discount,
            bsize=bsize,
            rmsize=rmsize,
            window_length=window_length,
            tau=tau,
            init_delta=init_delta,
            delta_decay=delta_decay,
            max_episode_length=max_episode_length,
            debug=debug,
            train_episode=train_episode,
            epsilon=epsilon
        )
        self.agent = DDPG(nb_states, nb_actions, self.ddpg_args)


    def compress(self):
        self.train(self.ddpg_args.train_episode, self.agent, self.env, self.output_dir)

    def train(self, num_episode, agent, env, output_dir):
        agent.is_training = True
        step = episode = episode_steps = 0
        episode_reward = 0.
        observation = None
        T = []  # trajectory
        while episode < num_episode:  # counting based on episode
            # reset if it is the start of episode
            if observation is None:
                observation = deepcopy(env.reset())
                agent.reset(observation)

            # agent pick action ...
            if episode <= self.ddpg_args.warmup:
                action = agent.random_action()
                # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
            else:
                action = agent.select_action(observation, episode=episode)

            # env response with next_observation, reward, terminate_info
            observation2, reward, done, info = env.step(action)

            T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

            # fix-length, never reach here
            # if max_episode_length and episode_steps >= max_episode_length - 1:
            #     done = True

            # [optional] save intermideate model
            if num_episode / 3 <= 1 or episode % int(num_episode / 3) == 0:
                agent.save_model(output_dir)

            # update
            step += 1
            episode_steps += 1
            episode_reward += reward
            observation = deepcopy(observation2)

            if done:  # end of episode
                _logger.info(
                    '#%d: episode_reward: %.4f acc: %.4f, ratio: %.4f',
                        episode, episode_reward,
                        info['accuracy'],
                        info['compress_ratio']
                )
                self.text_writer.write(
                    '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}\n'.format(
                        episode, episode_reward,
                        info['accuracy'],
                        info['compress_ratio']
                    )
                )
                final_reward = T[-1][0]
                # print('final_reward: {}'.format(final_reward))
                # agent observe and update policy
                for _, s_t, s_t1, a_t, done in T:
                    agent.observe(final_reward, s_t, s_t1, a_t, done)
                    if episode > self.ddpg_args.warmup:
                        agent.update_policy()

                #agent.memory.append(
                #    observation,
                #    agent.select_action(observation, episode=episode),
                #    0., False
                #)

                # reset
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
                T = []

                self.tfwriter.add_scalar('reward/last', final_reward, episode)
                self.tfwriter.add_scalar('reward/best', env.best_reward, episode)
                self.tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
                self.tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
                self.tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)
                # record the preserve rate for each layer
                for i, preserve_rate in enumerate(env.strategy):
                    self.tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

                self.text_writer.write('best reward: {}\n'.format(env.best_reward))
                self.text_writer.write('best policy: {}\n'.format(env.best_strategy))
        self.text_writer.close()
