# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from copy import deepcopy
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True

from tensorboardX import SummaryWriter

from nni.compression.torch.compressor import Pruner, LayerInfo, PrunerModuleWrapper
from .channel_pruning_env import ChannelPruningEnv
from .lib.agent import DDPG
from .lib.utils import get_output_folder

class AMCPruner(Pruner):
    """
    A pytorch implementation of AMC: AutoML for Model Compression and Acceleration on Mobile Devices.
    (https://arxiv.org/pdf/1802.03494.pdf)
    """
    def __init__(self, model, config_list, evaluator, val_loader, **kwargs):
        """
        Parameters:
            model: nn.Module
                The model to be pruned.
            config_list: list
                Configuration list to configure layer pruning.
                Supported keys:
                    op_types: operation type to be pruned
                    op_names: operation name to be pruned
            evaluator: function
                function to evaluate the pruned model.
                The prototype of the function:
                    >>> def val_func(val_loader, model):
                    >>>     ...
                    >>>     return acc
            val_loader: torch.utils.data.DataLoader
                Data loader of validation dataset.
            kwargs: dict
                Following additional parameters:
                job: str
                    'train' or 'export'
                suffix: str
                    suffix to help you remember what experiment you ran
                # environment
                model_type: str
                    model type to prune
                sparsity: float
                    prune ratio
                lbound: float
                    minimum prune ratio
                rbound: float
                    maximum prune ratio
                reward: function
                    reward function type
                # channel pruning
                n_calibration_batches: int
                    number of batches to extract layer information
                n_points_per_layer: int
                    number of feature points per layer
                channel_round: int
                    round channel to multiple of channel_round
                # ddpg
                hidden1: int
                    hidden num of first fully connect layer
                hidden2: int
                    hidden num of second fully connect layer
                lr_c: float
                    learning rate for critic
                lr_a: float
                    learning rate for actor
                warmup: int
                    time without training but only filling the replay memory
                discount: float
                bsize: int
                    minibatch size
                rmsize: int
                    memory size for each layer
                window_length: int
                tau: float
                    moving average for target network
                # noise
                init_delta: float
                    initial variance of truncated normal distribution
                delta_decay: float
                    delta decay during exploration
                # training
                max_episode_length: int
                    maximum episode length
                output: str
                    output dir
                debug: boolean
                    debug mode
                train_episode: int
                    train iters each timestep
                epsilon: int
                    linear decay of exploration policy
                seed: int
                    random seed to set
                # export
                ratios: list
                    ratios for pruning
                channels: list
                    channels after pruning
                export_path: str
                    path for exporting models
        """
        args = Namespace(**kwargs)
        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

        checkpoint = deepcopy(model.state_dict())

        super().__init__(model, config_list, optimizer=None)

        # convert sparsity to preserve ratio
        args.preserve_ratio = 1. - args.sparsity
        args.lbound, args.rbound = 1. - args.rbound, 1. - args.lbound

        self.env = ChannelPruningEnv(
            self, evaluator, val_loader, checkpoint,
            args.preserve_ratio, args=args)

        if args.job == 'train':
            # build folder and logs
            base_folder_name = '{}_{}_r{}_search'.format(args.model_type, args.dataset, args.preserve_ratio)
            if args.suffix is not None:
                base_folder_name = base_folder_name + '_' + args.suffix
            args.output = get_output_folder(args.output, base_folder_name)
            print('=> Saving logs to {}'.format(args.output))
            self.tfwriter = SummaryWriter(logdir=args.output)
            self.text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
            print('=> Output path: {}...'.format(args.output))

            nb_states = self.env.layer_embedding.shape[1]
            nb_actions = 1  # just 1 action here

            args.rmsize = args.rmsize * len(self.env.prunable_idx)  # for each layer
            print('** Actual replay buffer size: {}'.format(args.rmsize))

            self.agent = DDPG(nb_states, nb_actions, args)

        self.args = args

    def compress(self):
        if self.args.job == 'train':
            self.train(self.args.train_episode, self.agent, self.env, self.args.output)
        else:
            self.export()

    def train(self, num_episode, agent, env, output):
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
            if episode <= self.args.warmup:
                action = agent.random_action()
                # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
            else:
                action = agent.select_action(observation, episode=episode)

            # env response with next_observation, reward, terminate_info
            observation2, reward, done, info = env.step(action)
            observation2 = deepcopy(observation2)

            T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

            # fix-length, never reach here
            # if max_episode_length and episode_steps >= max_episode_length - 1:
            #     done = True

            # [optional] save intermideate model
            if episode % int(num_episode / 3) == 0:
                agent.save_model(output)

            # update
            step += 1
            episode_steps += 1
            episode_reward += reward
            observation = deepcopy(observation2)

            if done:  # end of episode
                print(
                    '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}'.format(
                        episode, episode_reward,
                        info['accuracy'],
                        info['compress_ratio']
                    )
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
                for r_t, s_t, s_t1, a_t, done in T:
                    agent.observe(final_reward, s_t, s_t1, a_t, done)
                    if episode > self.args.warmup:
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

    def export(self):
        wrapper_model_ckpt = 'best_wrapped_model.pth'
        self.bound_model.load_state_dict(torch.load(wrapper_model_ckpt))

        print('validate searched model:', self.env._validate(self.env._val_loader, self.env.model))
        self.env.export_model()
        self._unwrap_model()
        print('validate exported model:', self.env._validate(self.env._val_loader, self.env.model))
