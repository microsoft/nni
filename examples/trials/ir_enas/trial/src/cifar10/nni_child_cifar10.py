from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import fcntl
import numpy as np
import tensorflow as tf
import logging
import pickle
from src.utils import Logger
from src.cifar10.data_utils import read_data
from src.cifar10.general_child import GeneralChild
from src.cifar10.micro_child import MicroChild
from src.nni_child import ENASBaseTrial
from  src.cifar10_flags import *
import nni


def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


logger = build_logger("nni_child_cifar10")


def BuildChild(images, labels, ChildClass):
    child_model = ChildClass(
        images,
        labels,
        use_aux_heads=FLAGS.child_use_aux_heads,
        cutout_size=FLAGS.child_cutout_size,
        whole_channels=FLAGS.controller_search_whole_channels,
        num_layers=FLAGS.child_num_layers,
        num_cells=FLAGS.child_num_cells,
        num_branches=FLAGS.child_num_branches,
        fixed_arc=FLAGS.child_fixed_arc,
        out_filters_scale=FLAGS.child_out_filters_scale,
        out_filters=FLAGS.child_out_filters,
        keep_prob=FLAGS.child_keep_prob,
        drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
        num_epochs=FLAGS.num_epochs,
        l2_reg=FLAGS.child_l2_reg,
        data_format=FLAGS.data_format,
        batch_size=FLAGS.batch_size,
        clip_mode="norm",
        grad_bound=FLAGS.child_grad_bound,
        lr_init=FLAGS.child_lr,
        lr_dec_every=FLAGS.child_lr_dec_every,
        lr_dec_rate=FLAGS.child_lr_dec_rate,
        lr_cosine=FLAGS.child_lr_cosine,
        lr_max=FLAGS.child_lr_max,
        lr_min=FLAGS.child_lr_min,
        lr_T_0=FLAGS.child_lr_T_0,
        lr_T_mul=FLAGS.child_lr_T_mul,
        optim_algo="momentum",
        sync_replicas=FLAGS.child_sync_replicas,
        num_aggregate=FLAGS.child_num_aggregate,
        num_replicas=FLAGS.child_num_replicas,
    )

    return child_model


def get_child_ops(child_model):
    child_ops = {
        "global_step": child_model.global_step,
        "loss": child_model.loss,
        "train_op": child_model.train_op,
        "lr": child_model.lr,
        "grad_norm": child_model.grad_norm,
        "train_acc": child_model.train_acc,
        "optimizer": child_model.optimizer,
        "num_train_batches": child_model.num_train_batches,
        "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
        "eval_func": child_model.eval_once,
    }
    return child_ops


class ENASTrial(ENASBaseTrial):

    def __init__(self):

        if FLAGS.child_fixed_arc is None:
            images, labels = read_data(FLAGS.data_path)
        else:
            images, labels = read_data(FLAGS.data_path, num_valids=0)

        if FLAGS.search_for == "micro":
            ChildClass = MicroChild
        else:
            ChildClass = GeneralChild

        self.output_dir = os.path.join(os.getenv('NNI_OUTPUT_DIR'), '../..')
        self.file_path = os.path.join(self.output_dir, 'trainable_variable.txt')

        self.g = tf.Graph()
        with self.g.as_default():
            self.child_model = BuildChild(images, labels, ChildClass)

            self.total_data = {}

            self.child_model.connect_controller()
            self.child_model.build_valid_rl()
            self.child_ops = get_child_ops(self.child_model)

            self.saver = tf.train.Saver(max_to_keep=2)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

            hooks = []
            if FLAGS.child_sync_replicas:
                sync_replicas_hook = self.child_ops["optimizer"].make_session_run_hook(True)
                hooks.append(sync_replicas_hook)

            self.sess = tf.train.SingularMonitoredSession(
                config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir)

            self.load(self.file_path)
        logger.debug('initlize ENASTrial done.')

    def load(self, file_path):
        '''{variable_name: value}'''
        # first time, there's no file
        if not os.path.exists(file_path):
            return
        # otherwise load variable
        with open(file_path, 'rb') as fp:
            vals = pickle.load(fp)
        with self.g.as_default():
            for variable in tf.trainable_variables():
                name = variable.name
                if name in vals:
                    variable.load(vals[name], self.sess)

    def save(self, dir_path, file_path):
        if not os.path.exists(dir_path):
            os.mkdirs(dir_path)
        vals = dict()
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fp:
                vals = pickle.load(fp)
        with self.g.as_default():
            logger.debug(tf.trainable_variables()[:3])
            for variable in tf.trainable_variables():
                vals[variable.name] = self.sess.run(variable)
        with open(file_path, 'wb') as fp:
            pickle.dump(vals, fp)

    def get_child_arc_micro(self, controller_total_steps, normal_arc, reduce_arc):
        valid_acc_arr = []
        for idx in range(0, controller_total_steps):
            cur_valid_acc = self.sess.run(self.child_model.cur_valid_acc,
                                          feed_dict={self.child_model.normal_arc: normal_arc[idx],
                                                     self.child_model.reduce_arc: reduce_arc[idx]})
            valid_acc_arr.append(cur_valid_acc)
        return valid_acc_arr


    def parset_micro_arch(self,child_arc):

        normal_arc = []
        reduce_arc = []

        number = len(child_arc)
        half_number = number//2

        logger.debug("get arc total number\t"+str(half_number))

        for i in range(0,half_number):
            arc = child_arc[i]['__ndarray__']
            normal_arc.append(arc)

        for i in range(half_number,number):
            arc = child_arc[i]['__ndarray__']
            reduce_arc.append(arc)

        return normal_arc,reduce_arc


    def run_cchild_one_micro(self, child_totalsteps, normal_arc, reduce_arc):
        run_ops = [
            self.child_ops["loss"],
            self.child_ops["lr"],
            self.child_ops["grad_norm"],
            self.child_ops["train_acc"],
            self.child_ops["train_op"],
        ]

        actual_step = None
        epoch = None

        for step in range(0, child_totalsteps):

            loss, lr, gn, tr_acc, _ = self.sess.run\
                (run_ops, feed_dict={self.child_model.normal_arc: normal_arc[step],
                                     self.child_model.reduce_arc: reduce_arc[step]})

            global_step = self.sess.run(self.child_ops["global_step"])
            if FLAGS.child_sync_replicas:
                actual_step = global_step * FLAGS.num_aggregate
            else:
                actual_step = global_step
            epoch = actual_step // self.child_ops["num_train_batches"]
            if global_step % FLAGS.log_every == 0:
                log_string = ""
                log_string += "epoch={:<6d}".format(epoch)
                log_string += "ch_step={:<6d}".format(global_step)
                log_string += " loss={:<8.6f}".format(loss)
                log_string += " lr={:<8.4f}".format(lr)
                log_string += " |g|={:<8.4f}".format(gn)
                log_string += " tr_acc={:<3d}/{:>3d}".format(
                    tr_acc, FLAGS.batch_size)
                logger.debug(log_string)

        return actual_step, epoch


    def run_child_one_macro(self):
        run_ops = [
            self.child_ops["loss"],
            self.child_ops["lr"],
            self.child_ops["grad_norm"],
            self.child_ops["train_acc"],
            self.child_ops["train_op"],
        ]

        actual_step = None
        #epoch = None

        #for step in range(0, child_totalsteps):
        loss, lr, gn, tr_acc, _ = self.sess.run(run_ops)

        global_step = self.sess.run(self.child_ops["global_step"])

        # if FLAGS.child_sync_replicas:
        #     actual_step = global_step * FLAGS.num_aggregate
        # else:
        #     actual_step = global_step

        # epoch = actual_step // self.child_ops["num_train_batches"]
        log_string = ""
        #log_string += "epoch={:<6d}".format(epoch)
        log_string += "ch_step={:<6d}".format(global_step)
        log_string += " loss={:<8.6f}".format(loss)
        log_string += " lr={:<8.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " tr_acc={:<3d}/{:>3d}".format(
            tr_acc, FLAGS.batch_size)
        logger.debug(log_string)

        self.save(self.output_dir, self.file_path)
        return loss


    def start_eval_micro(self, first_arc):
        self.child_ops["eval_func"]\
            (self.sess, "valid", first_arc, self.child_model, SearchForMicro=True)
        self.child_ops["eval_func"]\
            (self.sess, "test", first_arc, self.child_model, SearchForMicro=True)


    def start_eval_macro(self, first_arc):
        self.child_ops["eval_func"]\
            (self.sess, "valid", first_arc, self.child_model, SearchForMicro=False)
        self.child_ops["eval_func"]\
            (self.sess, "test", first_arc, self.child_model, SearchForMicro=False)


def main(_):
    is_micro = False
    if FLAGS.search_for == "micro":
        is_micro = True
    logger.debug("-" * 80)

    if not os.path.isdir(FLAGS.output_dir):
        logger.debug("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        logger.debug("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
    logger.debug("-" * 80)
    '''@nni.get_next_parameter()'''
    trial = ENASTrial()
    controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
    logger.debug("here is the num train batches")

    logger.debug(trial.child_model.num_train_batches)
    child_totalsteps = (FLAGS.train_data_size + FLAGS.batch_size - 1) // FLAGS.batch_size
    logger.debug("child total \t"+str(child_totalsteps))
    epoch = 0

    """@nni.variable(nni.choice('train', 'validate'), name=entry)"""
    entry = 'trian'
    if is_micro:
        while True:
            if epoch >= FLAGS.num_epochs:
                break

            logger.debug("get parameter")
            #parameters =  nni.get_parameters()
            parameters = nni.get_next_parameter()
            logger.debug(parameters)

            normal_arc,reduce_arc = trial.parset_micro_arch(parameters)
            assert len(normal_arc) == len(reduce_arc)

            logger.debug("parse arch finish!")

            first_arc = (normal_arc[0], reduce_arc[0])
            logger.debug("len\t" + str(len(normal_arc)))

            actual_step, epoch = trial.run_cchild_one_micro(child_totalsteps, normal_arc, reduce_arc)

            logger.debug("epoch:\t" + str(epoch)+"actual_step:\t"+str(actual_step))
            valid_acc_arr = trial.get_child_arc_micro(controller_total_steps, normal_arc, reduce_arc)
            logger.debug("Get rewards Done!\n")

            nni.report_final_result(valid_acc_arr)
            logger.debug("Send rewards Done\n")

            trial.start_eval_micro(first_arc=first_arc)

    else:
        # while True:
        #     if epoch >= FLAGS.num_epochs:
        #         break

        logger.debug("get paramters")
        #child_arc = nni.get_parameters()
        # actual_step, epoch = trial.run_child_one_macro(child_totalsteps, child_arc)
        # logger.debug("epoch:\t" + str(epoch))

        
        if entry == 'train':
            loss = trial.run_child_one_macro()
            '''@nni.report_final_result(loss)'''
        elif entry == 'validate':
            valid_acc_arr = trial.get_csvaa(controller_total_steps, child_arc)
            '''@nni.report_final_result(valid_acc_arr)'''
            logger.debug("Get rewards Done!\n")
        else:
            raise RuntimeError('No such entry: ' + entry)

        logger.debug("Send rewards Done\n")
        #trial.start_eval_macro(first_arc=first_arc)


if __name__ == "__main__":
    tf.app.run()
