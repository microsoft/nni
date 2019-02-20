from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from src import utils
from src.utils import Logger
from src.utils import print_user_flags
from src.nni_child import ENASBaseTrial
from src.ptb.ptb_enas_child import PTBEnasChild
from src.nni_child import ENASBaseTrial
from src.ptb_flags import *
import nni


def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


logger = build_logger("nni_child_ptb")


def BuildChild(x_train, x_valid, x_test):
    child_model = PTBEnasChild(
        x_train,
        x_valid,
        x_test,
        rnn_l2_reg=FLAGS.child_rnn_l2_reg,
        rnn_slowness_reg=FLAGS.child_rnn_slowness_reg,
        rhn_depth=FLAGS.child_rhn_depth,
        fixed_arc=FLAGS.child_fixed_arc,
        batch_size=FLAGS.batch_size,
        bptt_steps=FLAGS.child_bptt_steps,
        lstm_num_layers=FLAGS.child_num_layers,
        lstm_hidden_size=FLAGS.child_lstm_hidden_size,
        lstm_e_keep=FLAGS.child_lstm_e_keep,
        lstm_x_keep=FLAGS.child_lstm_x_keep,
        lstm_h_keep=FLAGS.child_lstm_h_keep,
        lstm_o_keep=FLAGS.child_lstm_o_keep,
        lstm_l_skip=FLAGS.child_lstm_l_skip,
        vocab_size=10000,
        lr_init=FLAGS.child_lr,
        lr_dec_start=FLAGS.child_lr_dec_start,
        lr_dec_every=FLAGS.child_lr_dec_every,
        lr_dec_rate=FLAGS.child_lr_dec_rate,
        lr_dec_min=FLAGS.child_lr_dec_min,
        lr_warmup_val=FLAGS.child_lr_warmup_val,
        lr_warmup_steps=FLAGS.child_lr_warmup_steps,
        l2_reg=FLAGS.child_l2_reg,
        optim_moving_average=FLAGS.child_optim_moving_average,
        clip_mode="global",
        grad_bound=FLAGS.child_grad_bound,
        optim_algo="sgd",
        sync_replicas=FLAGS.child_sync_replicas,
        num_aggregate=FLAGS.child_num_aggregate,
        num_replicas=FLAGS.child_num_replicas,
        temperature=FLAGS.child_temperature,
        name="ptb_enas_model")
    return child_model


def get_child_ops(child_model):
    child_ops = {
        "global_step": child_model.global_step,
        "loss": child_model.loss,
        "train_op": child_model.train_op,
        "train_ppl": child_model.train_ppl,
        "train_reset": child_model.train_reset,
        "valid_reset": child_model.valid_reset,
        "test_reset": child_model.test_reset,
        "lr": child_model.lr,
        "grad_norm": child_model.grad_norm,
        "optimizer": child_model.optimizer,
        "num_train_batches": child_model.num_train_batches,
        "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
        "eval_func": child_model.eval_once,
    }
    return child_ops


class ENASTrial(ENASBaseTrial):

    def __init__(self):

        with open(FLAGS.data_path, "rb") as finp:
            x_train, x_valid, x_test, _, _ = pickle.load(finp,encoding="latin1")
            logger.debug("-" * 80)
            logger.debug("train_size: {0}".format(np.size(x_train)))
            logger.debug("valid_size: {0}".format(np.size(x_valid)))
            logger.debug(" test_size: {0}".format(np.size(x_test)))


        g = tf.Graph()
        with g.as_default():
            self.child_model = BuildChild(x_train, x_valid, x_test)
            self.child_model.connect_controller()

            self.child_ops = get_child_ops(self.child_model)

            self.saver = tf.train.Saver(max_to_keep=2)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
            hooks = []
            if FLAGS.child_sync_replicas:
                sync_replicas_hook = self.child_ops["optimizer"].make_session_run_hook(True)
                hooks.append(sync_replicas_hook)

            self.sess = tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir)
        logger.debug('initlize ENASTrial done.')


    def get_child_valid_loss(self, controller_total_steps, child_arc):
        valid_rl_loss_arr = []
        for idx in range(0, controller_total_steps):
            cur_rl_loss = self.sess.run(self.child_model.rl_loss, feed_dict={self.child_model.sample_arc: child_arc[idx]})
            valid_rl_loss_arr.append(cur_rl_loss)
        return valid_rl_loss_arr


    def ChildReset(self):
        self.sess.run([
            self.child_ops["train_reset"],
            self.child_ops["valid_reset"],
            self.child_ops["test_reset"],
        ])
        return


    def child_ooe_step(self, num_batches, total_tr_ppl, child_steps, child_arc):
        actual_step = None
        epoch = None
        for step in range(0, child_steps):
            run_ops = [
                self.child_ops["loss"],
                self.child_ops["lr"],
                self.child_ops["grad_norm"],
                self.child_ops["train_ppl"],
                self.child_ops["train_op"],
            ]
            loss, lr, gn, tr_ppl, _ = self.sess.run(run_ops,feed_dict={self.child_model.sample_arc:child_arc[step]})
            num_batches += 1
            total_tr_ppl += loss / FLAGS.child_bptt_steps
            global_step = self.sess.run(self.child_ops["global_step"])
            actual_step = global_step

            epoch = actual_step // self.child_ops["num_train_batches"]
            if global_step % FLAGS.log_every == 0:
                log_string = ""
                log_string += "epoch={:<6d}".format(epoch)
                log_string += " ch_step={:<6d}".format(global_step)
                log_string += " loss={:<8.4f}".format(loss)
                log_string += " lr={:<8.4f}".format(lr)
                log_string += " |g|={:<10.2f}".format(gn)
                log_string += " tr_ppl={:<8.2f}".format(
                    np.exp(total_tr_ppl / num_batches))
                logger.debug(log_string)

            if (FLAGS.child_reset_train_states is not None and
                    np.random.uniform(0, 1) < FLAGS.child_reset_train_states):
                logger.debug("reset train states")
                self.sess.run([
                    self.child_ops["train_reset"],
                    self.child_ops["valid_reset"],
                    self.child_ops["test_reset"],
                ])

            if actual_step % self.child_ops["eval_every"] == 0:
                self.sess.run([
                    self.child_ops["train_reset"],
                    self.child_ops["valid_reset"],
                    self.child_ops["test_reset"],
                ])

        return actual_step, epoch, num_batches, total_tr_ppl


def main(_):

    if not os.path.isdir(FLAGS.output_dir):
        logger.debug("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        logger.debug("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)

    logger.debug("-" * 80)

    logger.debug('Parse parameter done.')

    trial = ENASTrial()
    controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
    logger.debug("here is the num controller_total_steps\n")
    logger.debug(controller_total_steps)

    epoch = 0
    total_tr_ppl = 0.0
    num_batches = 0
    child_steps = FLAGS.child_steps
    logger.debug(child_steps)

    while True:
        if epoch >= FLAGS.num_epochs:
            break
        logger.debug("get paramters")
        #child_arc = nni.get_parameters()
        child_arc = nni.get_next_parameter()
        child_arc = trial.parset_child_arch(child_arc)

        first_arc = child_arc[0]
        logger.debug(first_arc)
        logger.debug("len\t" + str(len(child_arc)))
        actual_step, epoch,num_batches,total_tr_ppl = trial.child_ooe_step(num_batches, total_tr_ppl, child_steps, child_arc)
        trial.ChildReset()
        logger.debug("epoch:\t" + str(epoch))
        valid_rl_loss_arr = trial.get_child_valid_loss(controller_total_steps, child_arc)
        logger.debug("Get rl_loss Done!\n")
        nni.report_final_result(valid_rl_loss_arr)
        logger.debug("Send rewards Done\n")


if __name__ == "__main__":
  tf.app.run()