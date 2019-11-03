from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from src.common_ops import create_weight, batch_norm, batch_norm_with_mask, global_avg_pool, conv_op, pool_op
from src.utils import count_model_params, get_train_ops, get_C, get_strides
from src.cifar10.models import Model


class GeneralChild(Model):
    def __init__(self,
                 images,
                 labels,
                 cutout_size=None,
                 fixed_arc=None,
                 out_filters_scale=1,
                 num_layers=2,
                 num_branches=6,
                 out_filters=24,
                 keep_prob=1.0,
                 batch_size=32,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=0.1,
                 lr_dec_start=0,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_0=None,
                 lr_T_mul=None,
                 optim_algo=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="child",
                 mode="subgraph",
                 *args,
                 **kwargs
                 ):

        super(self.__class__, self).__init__(
            images,
            labels,
            cutout_size=cutout_size,
            batch_size=batch_size,
            clip_mode=clip_mode,
            grad_bound=grad_bound,
            l2_reg=l2_reg,
            lr_init=lr_init,
            lr_dec_start=lr_dec_start,
            lr_dec_every=lr_dec_every,
            lr_dec_rate=lr_dec_rate,
            keep_prob=keep_prob,
            optim_algo=optim_algo,
            sync_replicas=sync_replicas,
            num_aggregate=num_aggregate,
            num_replicas=num_replicas,
            data_format=data_format,
            name=name)

        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.out_filters = out_filters * out_filters_scale
        self.num_layers = num_layers
        self.mode = mode

        self.num_branches = num_branches
        self.fixed_arc = fixed_arc
        self.out_filters_scale = out_filters_scale

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]



    def _factorized_reduction(self, x, out_filters, stride, is_training):
        """Reduces the shape of x without information loss due to striding."""
        assert out_filters % 2 == 0, (
            "Need even number of filters when using this factorized reduction.")
        if stride == 1:
            with tf.variable_scope("path_conv"):
                inp_c = get_C(x, self.data_format)
                w = create_weight("w", [1, 1, inp_c, out_filters])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
                return x

        stride_spec = get_strides(stride, self.data_format)
        # Skip path 1
        path1 = tf.nn.avg_pool(
            x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path1_conv"):
            inp_c = get_C(path1, self.data_format)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format)

        # Skip path 2
        # First pad with 0"s on the right and bottom, then shift the filter to
        # include those 0"s that were added.
        if self.data_format == "NHWC":
            pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
            path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
            concat_axis = 3
        else:
            pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
            path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
            concat_axis = 1

        path2 = tf.nn.avg_pool(
            path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path2_conv"):
            inp_c = get_C(path2, self.data_format)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "SAME",
                                 data_format=self.data_format)

        # Concat and apply BN
        final_path = tf.concat(values=[path1, path2], axis=concat_axis)
        final_path = batch_norm(final_path, is_training,
                                data_format=self.data_format)

        return final_path

    def _model(self, images, is_training, reuse=False):
        '''Build model'''
        with tf.variable_scope(self.name, reuse=reuse):
            layers = []

            out_filters = self.out_filters
            with tf.variable_scope("stem_conv"):
                w = create_weight("w", [3, 3, 3, out_filters])
                x = tf.nn.conv2d(
                    images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
                layers.append(x)

            def add_fixed_pooling_layer(layer_id, layers, out_filters, is_training):
                '''Add a fixed pooling layer every four layers'''
                out_filters *= 2
                with tf.variable_scope("pool_at_{0}".format(layer_id)):
                    pooled_layers = []
                    for i, layer in enumerate(layers):
                        with tf.variable_scope("from_{0}".format(i)):
                            x = self._factorized_reduction(
                                layer, out_filters, 2, is_training)
                        pooled_layers.append(x)
                    return pooled_layers, out_filters

            def post_process_out(out, optional_inputs):
                '''Form skip connection and perform batch norm'''
                with tf.variable_scope("skip"):
                    inputs = layers[-1]
                    if self.data_format == "NHWC":
                        inp_h = inputs.get_shape()[1].value
                        inp_w = inputs.get_shape()[2].value
                        inp_c = inputs.get_shape()[3].value
                        out.set_shape([None, inp_h, inp_w, out_filters])
                    elif self.data_format == "NCHW":
                        inp_c = inputs.get_shape()[1].value
                        inp_h = inputs.get_shape()[2].value
                        inp_w = inputs.get_shape()[3].value
                        out.set_shape([None, out_filters, inp_h, inp_w])
                    optional_inputs.append(out)
                    pout = tf.add_n(optional_inputs)
                    out = batch_norm(pout, is_training,
                                     data_format=self.data_format)
                layers.append(out)
                return out

            global layer_id
            layer_id = -1

            def get_layer_id():
                global layer_id
                layer_id += 1
                return 'layer_' + str(layer_id)

            def conv3(inputs):
                # res_layers is pre_layers that are chosen to form skip connection
                # layers[-1] is always the latest input
                with tf.variable_scope(get_layer_id()):
                    with tf.variable_scope('branch_0'):
                        out = conv_op(
                            inputs[0][0], 3, is_training, out_filters, out_filters, self.data_format, start_idx=None)
                    out = post_process_out(out, inputs[1])
                return out

            def conv3_sep(inputs):
                with tf.variable_scope(get_layer_id()):
                    with tf.variable_scope('branch_1'):
                        out = conv_op(
                            inputs[0][0], 3, is_training, out_filters, out_filters, self.data_format, start_idx=None, separable=True)
                    out = post_process_out(out, inputs[1])
                return out

            def conv5(inputs):
                with tf.variable_scope(get_layer_id()):
                    with tf.variable_scope('branch_2'):
                        out = conv_op(
                            inputs[0][0], 5, is_training, out_filters, out_filters, self.data_format, start_idx=None)
                    out = post_process_out(out, inputs[1])
                return out

            def conv5_sep(inputs):
                with tf.variable_scope(get_layer_id()):
                    with tf.variable_scope('branch_3'):
                        out = conv_op(
                            inputs[0][0], 5, is_training, out_filters, out_filters, self.data_format, start_idx=None, separable=True)
                    out = post_process_out(out, inputs[1])
                return out

            def avg_pool(inputs):
                with tf.variable_scope(get_layer_id()):
                    with tf.variable_scope('branch_4'):
                        out = pool_op(
                            inputs[0][0], is_training, out_filters, out_filters, "avg", self.data_format, start_idx=None)
                    out = post_process_out(out, inputs[1])
                return out

            def max_pool(inputs):
                with tf.variable_scope(get_layer_id()):
                    with tf.variable_scope('branch_5'):
                        out = pool_op(
                            inputs[0][0], is_training, out_filters, out_filters, "max", self.data_format, start_idx=None)
                    out = post_process_out(out, inputs[1])
                return out

            """@nni.mutable_layers(
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs:[x],
                layer_output: layer_0_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs:[layer_0_out],
                optional_inputs: [layer_0_out],
                optional_input_size: [0, 1],
                layer_output: layer_1_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs:[layer_1_out],
                optional_inputs: [layer_0_out, layer_1_out],
                optional_input_size: [0, 1],
                layer_output: layer_2_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs:[layer_2_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out],
                optional_input_size: [0, 1],
                layer_output: layer_3_out
            }
            )"""
            layers, out_filters = add_fixed_pooling_layer(
                3, layers, out_filters, is_training)
            layer_0_out, layer_1_out, layer_2_out, layer_3_out = layers[-4:]
            """@nni.mutable_layers(
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs: [layer_3_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out],
                optional_input_size: [0, 1],
                layer_output: layer_4_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs: [layer_4_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out],
                optional_input_size: [0, 1],
                layer_output: layer_5_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs: [layer_5_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out],
                optional_input_size: [0, 1],
                layer_output: layer_6_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs: [layer_6_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out],
                optional_input_size: [0, 1],
                layer_output: layer_7_out
            }
            )"""
            layers, out_filters = add_fixed_pooling_layer(
                7, layers, out_filters, is_training)
            layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out = layers[
                -8:]
            """@nni.mutable_layers(
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs: [layer_7_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out],
                optional_input_size: [0, 1],
                layer_output: layer_8_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs: [layer_8_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out],
                optional_input_size: [0, 1],
                layer_output: layer_9_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs: [layer_9_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out],
                optional_input_size: [0, 1],
                layer_output: layer_10_out
            },
            {
                layer_choice: [conv3(), conv3_sep(), conv5(), conv5_sep(), avg_pool(), max_pool()],
                fixed_inputs:[layer_10_out],
                optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_10_out],
                optional_input_size: [0, 1],
                layer_output: layer_11_out
            }
            )"""

            x = global_avg_pool(layer_11_out, data_format=self.data_format)
            if is_training:
                x = tf.nn.dropout(x, self.keep_prob)
            with tf.variable_scope("fc"):
                if self.data_format == "NHWC":
                    inp_c = x.get_shape()[3].value
                elif self.data_format == "NCHW":
                    inp_c = x.get_shape()[1].value
                else:
                    raise ValueError(
                        "Unknown data_format {0}".format(self.data_format))
                w = create_weight("w", [inp_c, 10])
                x = tf.matmul(x, w)
        return x


    # override
    def _build_train(self):
        print("-" * 80)
        print("Build train graph")
        logits = self._model(self.x_train, is_training=True)
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.y_train)
        self.loss = tf.reduce_mean(log_probs)

        self.train_preds = tf.argmax(logits, axis=1)
        self.train_preds = tf.to_int32(self.train_preds)
        self.train_acc = tf.equal(self.train_preds, self.y_train)
        self.train_acc = tf.to_int32(self.train_acc)
        self.train_acc = tf.reduce_sum(self.train_acc)

        tf_variables = [var
                        for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.num_vars = count_model_params(tf_variables)
        print("Model has {} params".format(self.num_vars))

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step")

        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
            self.loss,
            tf_variables,
            self.global_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            lr_cosine=self.lr_cosine,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            lr_T_0=self.lr_T_0,
            lr_T_mul=self.lr_T_mul,
            num_train_batches=self.num_train_batches,
            optim_algo=self.optim_algo,
            sync_replicas=False,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas)

    # override
    def _build_valid(self):
        if self.x_valid is not None:
            print("-" * 80)
            print("Build valid graph")
            logits = self._model(self.x_valid, False, reuse=True)
            self.valid_preds = tf.argmax(logits, axis=1)
            self.valid_preds = tf.to_int32(self.valid_preds)
            self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
            self.valid_acc = tf.to_int32(self.valid_acc)
            self.valid_acc = tf.reduce_sum(self.valid_acc)

    # override
    def _build_test(self):
        print("-" * 80)
        print("Build test graph")
        logits = self._model(self.x_test, False, reuse=True)
        self.test_preds = tf.argmax(logits, axis=1)
        self.test_preds = tf.to_int32(self.test_preds)
        self.test_acc = tf.equal(self.test_preds, self.y_test)
        self.test_acc = tf.to_int32(self.test_acc)
        self.test_acc = tf.reduce_sum(self.test_acc)


    def build_model(self):

        self._build_train()
        self._build_valid()
        self._build_test()
