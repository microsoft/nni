from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
import logging
from src import utils
from src.utils import Logger
from src.utils import print_user_flags
from src.nni_controller import ENASBaseTuner
from src.ptb.ptb_enas_controller import PTBEnasController
from src.nni_controller import ENASBaseTuner
from src.ptb_flags import *


def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


logger = build_logger("nni_controller_ptb")


def BuildController():
    controller_model = PTBEnasController(
        rhn_depth=FLAGS.child_rhn_depth,
        lstm_size=100,
        lstm_num_layers=1,
        lstm_keep_prob=1.0,
        tanh_constant=FLAGS.controller_tanh_constant,
        temperature=FLAGS.controller_temperature,
        lr_init=FLAGS.controller_lr,
        lr_dec_start=0,
        lr_dec_every=1000000,  # never decrease learning rate
        l2_reg=FLAGS.controller_l2_reg,
        entropy_weight=FLAGS.controller_entropy_weight,
        bl_dec=FLAGS.controller_bl_dec,
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
        "valid_ppl": controller_model.valid_ppl,
        "optimizer": controller_model.optimizer,
        "baseline": controller_model.baseline,
        "ppl": controller_model.ppl,
        "reward": controller_model.reward,
        "entropy": controller_model.sample_entropy,
        "sample_arc": controller_model.sample_arc,
    }

    return controller_ops


class ENASTuner(ENASBaseTuner):

    def __init__(self, say_hello):
        logger.debug(say_hello)
        self.epoch = 0
        logger.debug('Parse parameter done.')
        self.controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
        logger.debug("controller_total_steps\n")
        logger.debug(self.controller_total_steps)
        self.child_steps = FLAGS.child_steps

        self.controller_model = BuildController()

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


    def receive_trial_result(self, parameter_id, parameters, reward, trial_job_id):
        logger.debug("epoch:\t"+str(self.epoch))
        logger.debug(parameter_id)
        logger.debug(reward)
        valid_acc_arr = reward
        self.controller_one_step(self.epoch, valid_acc_arr)
        return


    def controller_one_step(self, epoch, valid_loss_arr):
        logger.debug("Epoch {}: Training controller".format(epoch))
        for ct_step in range(self.controller_total_steps):
            run_ops = [
                self.controller_ops["loss"],
                self.controller_ops["entropy"],
                self.controller_ops["lr"],
                self.controller_ops["grad_norm"],
                self.controller_ops["reward"],
                self.controller_ops["baseline"],
                self.controller_ops["train_op"],
            ]
            loss, entropy, lr, gn, rw, bl, _ = self.sess.run(run_ops,feed_dict={self.controller_model.valid_loss:valid_loss_arr[ct_step]})
            controller_step = self.sess.run(self.controller_ops["train_step"])

            if ct_step % FLAGS.log_every == 0:
                log_string = ""
                log_string += "ctrl_step={:<6d}".format(controller_step)
                log_string += " loss={:<7.3f}".format(loss)
                log_string += " ent={:<5.2f}".format(entropy)
                log_string += " lr={:<6.4f}".format(lr)
                log_string += " |g|={:<10.7f}".format(gn)
                log_string += " rw={:<7.3f}".format(rw)
                log_string += " bl={:<7.3f}".format(bl)
                logger.debug(log_string)
        return


    def update_search_space(self, data):

        pass


    def generate_parameters(self, parameter_id, trial_job_id=None):
        child_arc = self.get_controller_arc_macro(self.child_steps)
        self.epoch = self.epoch + 1
        return child_arc


if __name__ == "__main__":
  tf.app.run()