# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# NNI (https://github.com/Microsoft/nni) modified this code to show how to
# integrate distributed tensorflow training with NNI SDK
#
"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import nni

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean(
    "download_only", False,
    "Only perform downloading of data; Do not proceed to "
    "session preparation, model definition or training")
flags.DEFINE_integer(
    "task_index", None, "Worker task index, should be >= 0. task_index=0 is "
    "the master worker task the performs the variable "
    "initialization ")
flags.DEFINE_integer(
    "num_gpus", 1, "Total number of gpus for each machine."
    "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer(
    "replicas_to_aggregate", None,
    "Number of replicas to aggregate before parameter update"
    "is applied (For sync_replicas mode only; default: "
    "num_workers)")
flags.DEFINE_integer("train_steps", 20000,
                     "Number of (global) training steps to perform")
flags.DEFINE_boolean(
    "sync_replicas", False,
    "Use the sync_replicas (synchronized replicas) mode, "
    "wherein the parameter updates from workers are aggregated "
    "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28

# Example:
#   cluster = {'ps': ['host1:2222', 'host2:2222'],
#              'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
#   os.environ['TF_CONFIG'] = json.dumps(
#       {'cluster': cluster,
#        'task': {'type': 'worker', 'index': 1}})


def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'learning_rate': 0.01,
        'batch_size': 100,
        'hidden_units': 100,
    }

def download_mnist_retry(data_dir, max_num_retries=20):
    """Try to download mnist dataset and avoid errors"""
    for _ in range(max_num_retries):
        try:
            return input_data.read_data_sets(data_dir, one_hot=True)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")

def main(unused_argv):
    # Receive NNI hyper parameter and update it onto default params
    RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = generate_default_params()
    PARAMS.update(RECEIVED_PARAMS)

    # Parse environment variable TF_CONFIG to get job_name and task_index

    # If not explicitly specified in the constructor and the TF_CONFIG
    # environment variable is present, load cluster_spec from TF_CONFIG.
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    task_config = tf_config.get('task', {})
    task_type = task_config.get('type')
    task_index = task_config.get('index')

    FLAGS.job_name = task_type
    FLAGS.task_index = task_index

    mnist = download_mnist_retry(FLAGS.data_dir)
    if FLAGS.download_only:
        sys.exit(0)

    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    cluster_config = tf_config.get('cluster', {})
    ps_hosts = cluster_config.get('ps')
    worker_hosts = cluster_config.get('worker')

    ps_hosts_str = ','.join(ps_hosts)
    worker_hosts_str = ','.join(worker_hosts)

    FLAGS.ps_hosts = ps_hosts_str
    FLAGS.worker_hosts = worker_hosts_str

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")

    # Get the number of workers.
    num_workers = len(worker_spec)

    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

    if not FLAGS.existing_servers:
        # Not using existing servers. Create an in-process server.
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()

    is_chief = (FLAGS.task_index == 0)
    if FLAGS.num_gpus > 0:
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        # Just allocate the CPU to worker server
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    with tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Variables of the hidden layer
        hid_w = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS * IMAGE_PIXELS, PARAMS['hidden_units']],
                stddev=1.0 / IMAGE_PIXELS),
            name="hid_w")
        hid_b = tf.Variable(tf.zeros([PARAMS['hidden_units']]), name="hid_b")

        # Variables of the softmax layer
        sm_w = tf.Variable(
            tf.truncated_normal(
                [PARAMS['hidden_units'], 10],
                stddev=1.0 / math.sqrt(PARAMS['hidden_units'])),
            name="sm_w")
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

        # Ops: located on the worker specified with FLAGS.task_index
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(
            y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        opt = tf.train.AdamOptimizer(PARAMS['learning_rate'])

        if FLAGS.sync_replicas:
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="mnist_sync_replicas")

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        if FLAGS.sync_replicas:
            local_init_op = opt.local_step_init_op
            if is_chief:
                local_init_op = opt.chief_init_op

            ready_for_local_init_op = opt.ready_for_local_init_op

            # Initial token and chief queue runners required by the sync_replicas mode
            chief_queue_runner = opt.get_chief_queue_runner()
            sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        if FLAGS.sync_replicas:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step)
        else:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=[
                "/job:ps", "/job:worker/task:%d" % FLAGS.task_index
            ])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        if FLAGS.existing_servers:
            server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
            print("Using existing server at: %s" % server_grpc_url)

            sess = sv.prepare_or_wait_for_session(
                server_grpc_url, config=sess_config)
        else:
            sess = sv.prepare_or_wait_for_session(
                server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        local_step = 0
        while True:
            # Training feed
            batch_xs, batch_ys = mnist.train.next_batch(PARAMS['batch_size'])
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print("%f: Worker %d: training step %d done (global step: %d)" %
                  (now, FLAGS.task_index, local_step, step))

            if step > 0 and step % 5000 == 0 and is_chief:
                val_feed = {
                    x: mnist.validation.images,
                    y_: mnist.validation.labels
                }
                interim_val_xent = sess.run(cross_entropy, feed_dict=val_feed)
                print(
                    "After %d training step(s), validation cross entropy = %g"
                    % (step, interim_val_xent))

                # Only chief worker can report intermediate metrics
                nni.report_intermediate_result(interim_val_xent)

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # Validation feed
        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print("After %d training step(s), validation cross entropy = %g" %
              (FLAGS.train_steps, val_xent))

        # Only chief worker can report final metrics
        if is_chief:
            nni.report_final_result(val_xent)


if __name__ == "__main__":
    tf.app.run()
