from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import logging
import tensorflow as tf
from src.cifar10.data_utils import read_data
from src.cifar10.general_child import GeneralChild
import src.cifar10_flags
from src.cifar10_flags import FLAGS


def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


logger = build_logger("nni_child_cifar10")


def build_trial(images, labels, ChildClass):
    '''Build child class'''
    child_model = ChildClass(
        images,
        labels,
        use_aux_heads=FLAGS.child_use_aux_heads,
        cutout_size=FLAGS.child_cutout_size,
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
        num_replicas=FLAGS.child_num_replicas
    )

    return child_model


def get_child_ops(child_model):
    '''Assemble child op to a dict'''
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


class NASTrial():

    def __init__(self):
        images, labels = read_data(FLAGS.data_path, num_valids=0)

        self.output_dir = os.path.join(os.getenv('NNI_OUTPUT_DIR'), '../..')
        self.file_path = os.path.join(
            self.output_dir, 'trainable_variable.txt')

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.child_model = build_trial(images, labels, GeneralChild)

            self.total_data = {}

            self.child_model.build_model()
            self.child_ops = get_child_ops(self.child_model)
            config = tf.ConfigProto(
                intra_op_parallelism_threads=0,
                inter_op_parallelism_threads=0,
                allow_soft_placement=True)

            self.sess = tf.train.SingularMonitoredSession(config=config)

        logger.debug('initlize NASTrial done.')

    def run_one_step(self):
        '''Run this model on a batch of data'''
        run_ops = [
            self.child_ops["loss"],
            self.child_ops["lr"],
            self.child_ops["grad_norm"],
            self.child_ops["train_acc"],
            self.child_ops["train_op"],
        ]
        loss, lr, gn, tr_acc, _ = self.sess.run(run_ops)
        global_step = self.sess.run(self.child_ops["global_step"])
        log_string = ""
        log_string += "ch_step={:<6d}".format(global_step)
        log_string += " loss={:<8.6f}".format(loss)
        log_string += " lr={:<8.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " tr_acc={:<3d}/{:>3d}".format(tr_acc, FLAGS.batch_size)
        if int(global_step) % FLAGS.log_every == 0:
            logger.debug(log_string)
        return loss, global_step

    def run(self):
        '''Run this model according to the `epoch` set in FALGS'''
        max_acc = 0
        while True:
            _, global_step = self.run_one_step()
            if global_step % self.child_ops['num_train_batches'] == 0:
                acc = self.child_ops["eval_func"](
                    self.sess, "test", self.child_model)
                max_acc = max(max_acc, acc)
                '''@nni.report_intermediate_result(acc)'''
            if global_step / self.child_ops['num_train_batches'] >= FLAGS.num_epochs:
                '''@nni.report_final_result(max_acc)'''
                break


def main(_):
    logger.debug("-" * 80)

    if not os.path.isdir(FLAGS.output_dir):
        logger.debug(
            "Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        logger.debug(
            "Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
    logger.debug("-" * 80)
    trial = NASTrial()

    trial.run()


if __name__ == "__main__":
    tf.app.run()
