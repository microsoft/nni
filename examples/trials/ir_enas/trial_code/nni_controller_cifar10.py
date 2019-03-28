from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import logging
import tensorflow as tf
import fcntl
import src.utils
import nni
from nni.multi_phase.multi_phase_tuner import MultiPhaseTuner
from src.utils import Logger
from src.cifar10.general_controller import GeneralController
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


def BuildController(ControllerClass):
    controller_model = ControllerClass(
        search_for=FLAGS.search_for,
        search_whole_channels=FLAGS.controller_search_whole_channels,
        skip_target=FLAGS.controller_skip_target,
        skip_weight=FLAGS.controller_skip_weight,
        num_cells=FLAGS.child_num_cells,
        num_layers=FLAGS.child_num_layers,
        num_branches=FLAGS.child_num_branches,
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
        num_replicas=FLAGS.controller_num_replicas)

    return controller_model


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


class ENASTuner(MultiPhaseTuner):

    def __init__(self, child_train_steps, controller_train_steps):
        # branches defaults to 6, need to be modified according to ss
        macro_init()

        # self.child_totalsteps = (FLAGS.train_data_size + FLAGS.batch_size - 1) // FLAGS.batch_size
        #self.controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
        self.child_train_steps = child_train_steps
        self.controller_train_steps = controller_train_steps
        self.total_steps = max(self.child_train_steps, self.controller_train_steps)
        logger.debug("child steps:\t"+str(self.child_train_steps))
        logger.debug("controller step\t"+str(self.controller_train_steps))

        ControllerClass = GeneralController
        self.controller_model = BuildController(ControllerClass)

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

        self.epoch = 0
        self.generate_one_epoch_parameters()

    def get_controller_arc_macro(self, child_totalsteps):
        child_arc = []
        for _ in range(0, child_totalsteps):
            arc = self.sess.run(self.controller_model.sample_arc)
            child_arc.append(arc)
        return child_arc

    def generate_one_epoch_parameters(self):
        # Generate architectures in one epoch and 
        # store them to self.child_arc
        self.pos = 0
        self.entry = 'train'
        self.child_arc = self.get_controller_arc_macro(self.total_steps)
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
        current_config = {self.key: self.entry}
        start_idx = 0
        onehot2list = lambda l: [idx for idx, val in enumerate(l) if val==1]
        for layer_id, (layer_name, info) in enumerate(self.search_space.items()):
            layer_choice_idx = current_arc_code[start_idx]
            if layer_id != 0:
                input_start = start_idx + 1
            else:
                input_start = start_idx
            inputs_idxs = current_arc_code[input_start: input_start + layer_id]
            inputs_idxs = onehot2list(inputs_idxs)
            current_config[layer_name] = dict()
            current_config[layer_name]['layer_choice'] = info['layer_choice'][layer_choice_idx]
            current_config[layer_name]['input_candidates'] = [info['input_candidates'][ipi] for ipi in inputs_idxs]
            start_idx += 1 + layer_id

        return current_config 


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


    def receive_trial_result(self, parameter_id, parameters, reward, trial_job_id):
        logger.debug("epoch:\t"+str(self.epoch))
        logger.debug(parameter_id)
        logger.debug(reward)
        if self.entry == 'validate':
            self.controller_one_step(self.epoch, reward)

    def update_search_space(self, data):
        # Extract choice
        self.key = list(filter(lambda k: k.strip().endswith('choice'), list(data)))[0]
        data.pop(self.key)
        # Sort layers
        self.search_space = OrderedDict(sorted(data.items(), key=lambda tp:int(tp[0].split('_')[1])))
        logger.debug(self.search_space)

if __name__ == "__main__":
    tf.app.run()
