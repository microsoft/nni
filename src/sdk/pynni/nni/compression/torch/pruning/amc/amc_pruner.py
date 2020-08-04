# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from copy import deepcopy
from argparse import Namespace
import numpy as np
import torch
torch.backends.cudnn.deterministic = True

from tensorboardX import SummaryWriter

from nni.compression.torch.compressor import Pruner
from .channel_pruning_env import ChannelPruningEnv
from .lib.agent import DDPG
from .lib.utils import get_output_folder

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
        suffix: str
            suffix to help you remember what experiment you ran
        job: str
            train: search best pruned model.
            export: export a searched model, before exporting model, you must run this pruner
            to search a pruned model with job=train.
        export_path: str
            path for exporting models

        # parameters for pruning environment
        model_type: str
            model type to prune, currently 'mobilenet' and 'mobilenetv2' are supported.
        sparsity: float
            prune ratio
        lbound: float
            minimum prune ratio
        rbound: float
            maximum prune ratio
        reward: function
            reward function type

        # parameters for channel pruning
        n_calibration_batches: int
            number of batches to extract layer information
        n_points_per_layer: int
            number of feature points per layer
        channel_round: int
            round channel to multiple of channel_round

        # parameters for ddpg agent
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

        # parameters for training ddpg agent
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
    """

    def __init__(
            self,
            model,
            config_list,
            evaluator,
            val_loader,
            suffix=None,
            job='train',
            export_path=None,
            model_type='mobilenet',
            sparsity=0.5,
            lbound=0.,
            rbound=0.8,
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
            delta_decay=0.95,
            max_episode_length=1e9,
            output='./logs',
            debug=False,
            train_episode=800,
            epsilon=50000,
            seed=None):

        self.job = job
        self.export_path = export_path

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        checkpoint = deepcopy(model.state_dict())

        super().__init__(model, config_list, optimizer=None)

        # convert sparsity to preserve ratio
        preserve_ratio = 1. - sparsity
        lbound, rbound = 1. - rbound, 1. - lbound

        self.env_args = Namespace(
            model_type=model_type,
            preserve_ratio=preserve_ratio,
            lbound=lbound,
            rbound=rbound,
            reward=reward,
            n_calibration_batches=n_calibration_batches,
            n_points_per_layer=n_points_per_layer,
            channel_round=channel_round
        )

        self.env = ChannelPruningEnv(
            self, evaluator, val_loader, checkpoint, args=self.env_args)

        if self.job == 'train':
            # build folder and logs
            base_folder_name = '{}_r{}_search'.format(model_type, preserve_ratio)
            if suffix is not None:
                base_folder_name = base_folder_name + '_' + suffix
            self.output = get_output_folder(output, base_folder_name)
            print('=> Saving logs to {}'.format(self.output))
            self.tfwriter = SummaryWriter(logdir=self.output)
            self.text_writer = open(os.path.join(self.output, 'log.txt'), 'w')
            print('=> Output path: {}...'.format(self.output))

            nb_states = self.env.layer_embedding.shape[1]
            nb_actions = 1  # just 1 action here

            rmsize = rmsize * len(self.env.prunable_idx)  # for each layer
            print('** Actual replay buffer size: {}'.format(rmsize))

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
        if self.job == 'train':
            self.train(self.ddpg_args.train_episode, self.agent, self.env, self.output)
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
            if episode <= self.ddpg_args.warmup:
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
            if num_episode / 3 <= 1 or episode % int(num_episode / 3) == 0:
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

    def export(self):
        wrapper_model_ckpt = 'best_wrapped_model.pth'
        self.bound_model.load_state_dict(torch.load(wrapper_model_ckpt))

        print('validate searched model:', self.env._validate(self.env._val_loader, self.env.model))
        self.env.export_model()
        self._unwrap_model()
        print('validate exported model:', self.env._validate(self.env._val_loader, self.env.model))

        torch.save(self.bound_model, self.export_path)
        print('exported model saved to: {}'.format(self.export_path))
