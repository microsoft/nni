# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from copy import deepcopy

import numpy as np
import torch
torch.backends.cudnn.deterministic = True

from tensorboardX import SummaryWriter

from nni.compression.torch.compressor import Pruner, LayerInfo, PrunerModuleWrapper
from .channel_pruning_env import ChannelPruningEnv
#from .channel_pruning_env_orig import ChannelPruningEnv
from .lib.agent import DDPG
from .lib.utils import get_output_folder

class AMCPruner(Pruner):
    def __init__(self, model, config_list, val_func, val_loader, args):        
        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
        
        checkpoint = deepcopy(model.state_dict())

        super().__init__(model, config_list, optimizer=None)


        self.env = ChannelPruningEnv(self, val_func, val_loader, checkpoint,
                                preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                                args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)

        #self.env = ChannelPruningEnv(model, val_func, val_loader,
        #                        preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
        #                        args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)


        if args.job == 'train':
            # build folder and logs
            base_folder_name = '{}_{}_r{}_search'.format(args.model, args.dataset, args.preserve_ratio)
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
            #train(args.train_episode, agent, env, args.output)

        self.args = args

    def compress(self):
        self.train(self.args.train_episode, self.agent, self.env, self.args.output)

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
                print('#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}'.format(episode, episode_reward,
                                                                                    info['accuracy'],
                                                                                    info['compress_ratio']))
                self.text_writer.write(
                    '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}\n'.format(episode, episode_reward,
                                                                                    info['accuracy'],
                                                                                    info['compress_ratio']))
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
        assert self.args.ratios is not None or self.args.channels is not None, 'Please provide a valid ratio list or pruned channels'
        assert self.args.export_path is not None, 'Please provide a valid export path'
        self.env.set_export_path(self.args.export_path)

        print('=> Original model channels: {}'.format(self.env.org_channels))
        if self.args.ratios:
            ratios = self.args.ratios.split(',')
            ratios = [float(r) for r in ratios]
            assert  len(ratios) == len(self.env.org_channels)
            channels = [int(r * c) for r, c in zip(ratios, self.env.org_channels)]
        else:
            channels = self.args.channels.split(',')
            channels = [int(r) for r in channels]
            ratios = [c2 / c1 for c2, c1 in zip(channels, self.env.org_channels)]
        print('=> Pruning with ratios: {}'.format(ratios))
        print('=> Channels after pruning: {}'.format(channels))

        for r in ratios:
            self.env.step(r)

        return
