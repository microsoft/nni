from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import shutil
import sys
import time
import logging
import tensorflow as tf
import fcntl
import src.utils
import nni
from src.utils import Logger
from src.cifar10.general_controller import GeneralController
from src.cifar10.micro_controller import MicroController
from src.nni_controller import ENASBaseTuner
from src.cifar10_flags import *
from collections import OrderedDict

def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


logger = build_logger("nni_controller_cifar10")





def get_controller_ops(controller_model):
    """
    Args:
      images: dict with keys {"train", "valid", "test"}.
      labels: dict with keys {"train", "valid", "test"}.
    """

    controller_ops = {
        "train_step": controller_model.train_step,
        "loss": controller_model.loss,
        "train_op": controller_model.train_op,
        "lr": controller_model.lr,
        "grad_norm": controller_model.grad_norm,
        "valid_acc": controller_model.valid_acc,
        "optimizer": controller_model.optimizer,
        "baseline": controller_model.baseline,
        "entropy": controller_model.sample_entropy,
        "sample_arc": controller_model.sample_arc,
        "skip_rate": controller_model.skip_rate,
    }

    return controller_ops


class ENASTuner(ENASBaseTuner):

    def __init__(self, child_train_steps, controller_train_steps, macro_str="macro"):
        # branches defaults to 6, need to be modified according to ss
        if macro_str == "macro":
            self.Is_macro = True
            macro_init()
        else:
            self.Is_macro = False
            micro_init()

        logger.debug('Parse parameter done.')
        logger.debug(macro_str)

        # self.child_totalsteps = (FLAGS.train_data_size + FLAGS.batch_size - 1) // FLAGS.batch_size

        #self.controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
        self.child_train_steps = child_train_steps
        self.controller_train_steps = controller_train_steps
        self.total_steps = max(self.child_train_steps, self.controller_train_steps)
        logger.debug("child steps:\t"+str(self.child_train_steps))
        logger.debug("controller step\t"+str(self.controller_train_steps))

        

        
    def init_controller(self):
        if FLAGS.search_for == "micro":
            ControllerClass = MicroController
        else:
            ControllerClass = GeneralController
        self.controller_model = self.BuildController(ControllerClass)

        self.graph = tf.Graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        self.controller_model.build_trainer()
        self.controller_ops = get_controller_ops(self.controller_model)

        hooks = []
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
            sync_replicas_hook = self.controller_ops["optimizer"].make_session_run_hook(True)
            hooks.append(sync_replicas_hook)

        self.sess = tf.train.SingularMonitoredSession(
            config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir)
        logger.debug('initlize controller_model done.')

        self.generate_one_epoch_parameters()
        self.entry = 'train'
        self.pos = 0
        self.epoch = 0

    def generate_one_epoch_parameters(self):
        # Generate architectures in one epoch and 
        # store them to self.child_arc
        self.pos = 0
        self.entry = 'train'
        if self.Is_macro:
            self.child_arc = self.get_controller_arc_macro(self.total_steps)
            self.epoch = self.epoch + 1
        else:
            normal_arc,reduce_arc = self.get_controller_arc_micro(self.total_steps)
            self.result_arc = normal_arc
            self.result_arc.extend(reduce_arc)
            self.epoch = self.epoch + 1    


    def generate_parameters(self, parameter_id, trial_job_id=None):
        self.pos += 1
        logger.info('current pos: ' + str(self.pos))
        if self.pos == self.child_train_steps + 1:
            self.entry = 'validate'
        elif self.pos > self.child_train_steps + self.controller_train_steps:
            self.generate_one_epoch_parameters()

        if len(self.child_arc) <= 0:
            raise nni.NoMoreTrialError('no more parameters now.')

        current_arc_code = self.child_arc[self.pos - (1 if self.entry=='train' else (self.child_train_steps+1))]
        logger.debug('current_arc_code: ' + str(current_arc_code))
        current_config = {self.key: self.entry}
        start_idx = 0
        onehot2list = lambda l: [idx for idx, val in enumerate(l) if val==1]
        for layer_id, (layer_name, info) in enumerate(self.search_space.items()):
            layer_choice_idx = current_arc_code[start_idx]
            if layer_id != 0:
                input_start = start_idx + 1
            else:
                input_start = start_idx
            num_input_candidates = len(info['input_candidates'])
            inputs_idxs = current_arc_code[input_start: input_start + num_input_candidates]
            inputs_idxs = onehot2list(inputs_idxs)
            current_config[layer_name] = dict()
            current_config[layer_name]['layer_choice'] = info['layer_choice'][layer_choice_idx]
            current_config[layer_name]['input_candidates'] = [info['input_candidates'][ipi] for ipi in inputs_idxs]
            start_idx += 1 + num_input_candidates

        return current_config 


    def get_controller_arc_micro(self, child_totalsteps):
        normal_arc = []
        reduce_arc = []
        for _ in range(0, child_totalsteps):
            arc1, arc2 = self.sess.run(self.controller_model.sample_arc)
            normal_arc.append(arc1)
            reduce_arc.append(arc2)
        return normal_arc,reduce_arc


    def controller_one_step(self, epoch, valid_acc_arr):
        logger.debug("Epoch {}: Training controller".format(epoch))

        #for ct_step in range(FLAGS.controller_train_steps * FLAGS.controller_num_aggregate):
        run_ops = [
            self.controller_ops["loss"],
            self.controller_ops["entropy"],
            self.controller_ops["lr"],
            self.controller_ops["grad_norm"],
            self.controller_ops["valid_acc"],
            self.controller_ops["baseline"],
            self.controller_ops["skip_rate"],
            self.controller_ops["train_op"],
        ]

        loss, entropy, lr, gn, val_acc, bl, _, _ = self.sess.run(run_ops, feed_dict={
            self.controller_model.valid_acc: valid_acc_arr})

        controller_step = self.sess.run(self.controller_ops["train_step"])

        log_string = ""
        log_string += "ctrl_step={:<6d}".format(controller_step)
        log_string += " loss={:<7.3f}".format(loss)
        log_string += " ent={:<5.2f}".format(entropy)
        log_string += " lr={:<6.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " acc={:<6.4f}".format(val_acc)
        log_string += " bl={:<5.2f}".format(bl)
        log_string += " child acc={:<5.2f}".format(valid_acc_arr)
        logger.debug(log_string)
        return


    def receive_trial_result(self, parameter_id, parameters, reward):
        logger.debug("epoch:\t"+str(self.epoch))
        logger.debug(parameter_id)
        logger.debug(reward)
        if self.entry == 'validate':
            self.controller_one_step(self.epoch, reward)
        return


    def update_search_space(self, data):
        # Extract choice
        self.key = list(filter(lambda k: k.strip().endswith('choice'), list(data)))[0]
        data.pop(self.key)
        # Sort layers
        self.search_space = OrderedDict(sorted(data.items(), key=lambda tp:int(tp[0].split('_')[1])))
        logger.debug(self.search_space)
        # Number each layer_choice and outputs and input_candidate
        num_branches = 0
        self.branches = dict()
        self.outputs = dict()
        self.hash_search_space = list()
        for layer_id, (_, info) in enumerate(self.search_space.items()):
            hash_info = {'layer_choice': []}
            # record branch_name <--> branch_id
            for branch_idx in range(len(info['layer_choice'])):
                branch_name = info['layer_choice'][branch_idx]
                if branch_name not in self.branches:
                    self.branches[branch_name] = num_branches
                    hash_info['layer_choice'].append(num_branches)
                    num_branches += 1
                else:
                    hash_info['layer_choice'].append(self.branches[branch_name])
            assert info['outputs'] not in self.outputs, 'Output variables from different layers cannot be the same'
            # record output_name <--> output_id
            self.outputs[layer_id], self.outputs[info['outputs']] = info['outputs'], layer_id
            # convert input_candidate to id
            if layer_id != 0:
                hash_info['input_candidates'] = list()
                for candidate in info['input_candidates']:
                    assert candidate in self.outputs, 'Subsequent layers must use the output of the previous layer as an input candidate'
                    hash_info['input_candidates'].append(self.outputs[candidate])
            self.hash_search_space.append(hash_info)
        logger.debug(self.hash_search_space)
        self.init_controller()
        

    def BuildController(self, ControllerClass):
        controller_model = ControllerClass(
            search_for=FLAGS.search_for,
            search_whole_channels=FLAGS.controller_search_whole_channels,
            skip_target=FLAGS.controller_skip_target,
            skip_weight=FLAGS.controller_skip_weight,
            num_cells=FLAGS.child_num_cells,
            out_filters=FLAGS.child_out_filters,
            lstm_size=64,
            lstm_num_layers=1,
            lstm_keep_prob=1.0,
            tanh_constant=FLAGS.controller_tanh_constant,
            op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
            temperature=FLAGS.controller_temperature,
            lr_init=FLAGS.controller_lr,
            lr_dec_start=0,
            lr_dec_every=1000000,  # never decrease learning rate
            l2_reg=FLAGS.controller_l2_reg,
            entropy_weight=FLAGS.controller_entropy_weight,
            bl_dec=FLAGS.controller_bl_dec,
            use_critic=FLAGS.controller_use_critic,
            optim_algo="adam",
            sync_replicas=FLAGS.controller_sync_replicas,
            num_aggregate=FLAGS.controller_num_aggregate,
            num_replicas=FLAGS.controller_num_replicas,
            num_layers=FLAGS.child_num_layers,
            num_branches=len(self.branches),
            hash_search_space=self.hash_search_space)

        return controller_model

if __name__ == "__main__":
    tf.app.run()
