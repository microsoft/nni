# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import logging

from .. import trial


_logger = logging.getLogger(__name__)
_MUTABLE_LAYER_SPACE_PREFIX = "_mutable_layer"
_namespace = {}
_tf_variables = {}
_arch_logits_list = []
_optimizer = None
_train_op = None


def classic_mode(
        mutable_id,
        mutable_layer_id,
        funcs,
        funcs_args,
        fixed_inputs,
        optional_inputs,
        optional_input_size):
    '''Execute the chosen function and inputs directly.
    In this mode, the trial code is only running the chosen subgraph (i.e., the chosen ops and inputs),
    without touching the full model graph.'''
    if trial.get_current_parameter() is None:
        trial.get_next_parameter()

    chosen_layer, chosen_inputs = _get_layer_and_inputs_from_tuner(mutable_id, mutable_layer_id,
                                                                   list(optional_inputs.keys()))
    real_chosen_inputs = [optional_inputs[input_name] for input_name in chosen_inputs]
    layer_out = funcs[chosen_layer]([fixed_inputs, real_chosen_inputs], **funcs_args[chosen_layer])

    return layer_out


def enas_mode(
        mutable_id,
        mutable_layer_id,
        funcs,
        funcs_args,
        fixed_inputs,
        optional_inputs,
        optional_input_size,
        tf):
    '''For enas mode, we build the full model graph in trial but only run a subgraph。
    This is implemented by masking inputs and branching ops.
    Specifically, based on the received subgraph (through nni.get_next_parameter),
    it can be known which inputs should be masked and which op should be executed.'''
    name_prefix = "{}_{}".format(mutable_id, mutable_layer_id)
    # store namespace
    _namespace[mutable_id] = True
    _namespace[name_prefix] = dict()
    _namespace[name_prefix]['funcs'] = list(funcs)
    _namespace[name_prefix]['optional_inputs'] = list(optional_inputs)
    # create tensorflow variables as 1/0 signals used to form subgraph
    name_for_optional_inputs = name_prefix + '_optional_inputs'
    name_for_funcs = name_prefix + '_funcs'
    _tf_variables[name_prefix] = dict()
    _tf_variables[name_prefix]['optional_inputs'] = tf.get_variable(
        name_for_optional_inputs,
        [len(optional_inputs)],
        dtype=tf.bool,
        trainable=False
    )
    _tf_variables[name_prefix]['funcs'] = tf.get_variable(
        name_for_funcs, [], dtype=tf.int64, trainable=False)

    # get real values using their variable names
    real_optional_inputs_value = [optional_inputs[name]
                                  for name in _namespace[name_prefix]['optional_inputs']]
    real_func_value = [funcs[name]
                       for name in _namespace[name_prefix]['funcs']]
    real_funcs_args = [funcs_args[name]
                       for name in _namespace[name_prefix]['funcs']]
    # build tensorflow graph of geting chosen inputs by masking
    real_chosen_inputs = tf.boolean_mask(
        real_optional_inputs_value, _tf_variables[name_prefix]['optional_inputs'])
    # build tensorflow graph of different branches by using tf.case
    branches = dict()
    func_output = None
    for func_id in range(len(funcs)):
        func_output = real_func_value[func_id]([fixed_inputs, real_chosen_inputs], **real_funcs_args[func_id])
        branches[tf.equal(_tf_variables[name_prefix]['funcs'], func_id)] = lambda: func_output
    layer_out = tf.case(branches, exclusive=True, default=lambda: func_output)

    return layer_out


def oneshot_mode(
        mutable_id,
        mutable_layer_id,
        funcs,
        funcs_args,
        fixed_inputs,
        optional_inputs,
        optional_input_size,
        tf):
    '''Similar to enas mode, oneshot mode also builds the full model graph.
    The difference is that oneshot mode does not receive subgraph.
    Instead, it uses dropout to randomly dropout inputs and ops.'''
    # NNI requires to get_next_parameter before report a result. But the parameter will not be used in this mode
    if trial.get_current_parameter() is None:
        trial.get_next_parameter()
    optional_inputs = list(optional_inputs.values())
    inputs_num = len(optional_inputs)
    # Calculate dropout rate according to the formular r^(1/k), where r is a hyper-parameter and k is the number of inputs
    if inputs_num > 0:
        rate = 0.01 ** (1 / inputs_num)
        noise_shape = [inputs_num] + [1] * len(optional_inputs[0].get_shape())
        optional_inputs = tf.nn.dropout(
            optional_inputs, rate=rate, noise_shape=noise_shape)
        optional_inputs = [optional_inputs[idx] for idx in range(inputs_num)]
    layer_outs = [func([fixed_inputs, optional_inputs], **funcs_args[func_name])
                  for func_name, func in funcs.items()]
    output_num = len(layer_outs)
    rate = 0.01 ** (1 / output_num)
    noise_shape = [output_num] + [1] * len(layer_outs[0].get_shape())
    layer_outs = tf.nn.dropout(layer_outs, rate=rate, noise_shape=noise_shape)
    layer_out = tf.reduce_sum(layer_outs, axis=0)

    return layer_out


def darts_mode(
        mutable_id,
        mutable_layer_id,
        funcs,
        funcs_args,
        fixed_inputs,
        optional_inputs,
        optional_input_size,
        tf):
    optional_inputs = list(optional_inputs.values())
    layer_outs = [func([fixed_inputs, optional_inputs], **funcs_args[func_name])
                  for func_name, func in funcs.items()]
    # Create architecture weights for every func(op)
    var_name = "{}_{}_arch_weights".format(mutable_id, mutable_layer_id)
    arch_logits = tf.get_variable(var_name, shape=[len(funcs)], trainable=False)
    _arch_logits_list.append(arch_logits)
    arch_weights = tf.nn.softmax(arch_logits)
    layer_out = tf.add_n([arch_weights[idx] * out for idx, out in enumerate(layer_outs)])

    return layer_out


def reload_tensorflow_variables(tf, session):
    '''In Enas mode, this function reload every signal varaible created in `enas_mode` function so
    the whole tensorflow graph will be changed into certain subgraph recerived from Tuner.
    ---------------
    session: the tensorflow session created by users
    tf: tensorflow module
    '''
    subgraph_from_tuner = trial.get_next_parameter()
    mutable_layers = set()
    for subgraph_key in subgraph_from_tuner:
        if "/" in subgraph_key:
            # has to remove the last, could be layer_choice or whatever
            mutable_id, mutable_layer_id = _decompose_general_key(subgraph_key[:subgraph_key.rfind("/")])
            if mutable_id is not None:
                mutable_layers.add((mutable_id, mutable_layer_id))
    mutable_layers = sorted(list(mutable_layers))
    for mutable_id, mutable_layer_id in mutable_layers:
        if mutable_id not in _namespace:
            _logger.warning("%s not found in name space", mutable_id)
            continue
        name_prefix = "{}_{}".format(mutable_id, mutable_layer_id)
        # get optional inputs names
        optional_inputs = _namespace[name_prefix]['optional_inputs']
        # extract layer information from the subgraph sampled by tuner
        chosen_layer, chosen_inputs = _get_layer_and_inputs_from_tuner(mutable_id, mutable_layer_id, optional_inputs)
        chosen_layer = _namespace[name_prefix]['funcs'].index(chosen_layer)
        chosen_inputs = [1 if inp in chosen_inputs else 0 for inp in optional_inputs]
        # load these information into pre-defined tensorflow variables
        _tf_variables[name_prefix]['funcs'].load(chosen_layer, session)
        _tf_variables[name_prefix]['optional_inputs'].load(
            chosen_inputs, session)


def _construct_general_key(mutable_id, mutable_layer_id):
    # Mutable layer key in a general (search space) format
    # that is, prefix/mutable_id/mutable_layer_id
    return _MUTABLE_LAYER_SPACE_PREFIX + "/" + mutable_id + "/" + mutable_layer_id


def _decompose_general_key(key):
    # inverse operation of above
    if not key.startswith(_MUTABLE_LAYER_SPACE_PREFIX):
        return None, None
    else:
        _, mutable_id, mutable_layer_id = key.split("/", maxsplit=2)
        return mutable_id, mutable_layer_id


def darts_training(tf, session, loss, feed_dict):
    global _optimizer, _train_op
    if _optimizer is None:
        _optimizer = tf.MomentumOptimizer(learning_rate=0.025)
        # TODO: Calculate loss
        grads_and_vars = _optimizer.compute_gradients(loss, _arch_logits_list)
        _train_op = _optimizer.apply_gradients(grads_and_vars)
    session.run(_train_op)


def training_update(nas_mode, tf=None, session=None, loss=None, feed_dict=None):
    if nas_mode == 'darts_mode':
        darts_training(tf, session, loss, feed_dict)
    elif nas_mode == 'enas_mode':
        reload_tensorflow_variables(tf, session)


def _get_layer_and_inputs_from_tuner(mutable_id, mutable_layer_id, optional_inputs):
    # optional_inputs should be name(key)s of the optional inputs
    try:
        mutable_block = trial.get_current_parameter(mutable_id)

        # There is a NAS tuner
        chosen_layer = mutable_block[mutable_layer_id]["chosen_layer"]
        chosen_inputs = mutable_block[mutable_layer_id]["chosen_inputs"]
    except KeyError:
        # Try to find converted NAS parameters
        params = trial.get_current_parameter()
        expected_prefix = _construct_general_key(mutable_id, mutable_layer_id)
        chosen_layer = params[expected_prefix + "/layer_choice"]

        # find how many to choose
        optional_input_size = int(params[expected_prefix + "/optional_input_size"])  # convert uniform to randint

        # find who to choose, can duplicate
        optional_input_state = params[expected_prefix + "/optional_input_chosen_state"]
        chosen_inputs = []
        # make sure dict -> list produce stable result by sorting
        optional_inputs_keys = sorted(optional_inputs)
        for _ in range(optional_input_size):
            chosen_inputs.append(optional_inputs_keys[optional_input_state % len(optional_inputs)])
            optional_input_state //= len(optional_inputs)

    _logger.info("%s_%s: layer: %s, optional inputs: %s", mutable_id, mutable_layer_id, chosen_layer, chosen_inputs)
    return chosen_layer, chosen_inputs


def convert_nas_search_space(search_space):
    """
    Args:
        param search_space: raw search space
        return: the new search space, mutable_layers will be converted into choice
    """
    if not isinstance(search_space, dict):
        return search_space
    ret = dict()
    for k, v in search_space.items():
        if "_type" not in v:
            # this should not happen
            _logger.warning("There is no _type in one of your search space values with key '%s'"
                            ". Please check your search space", k)
            ret[k] = v
        elif v["_type"] != "mutable_layer":
            ret[k] = v
        else:
            _logger.info("Converting mutable_layer search space with key '%s'", k)
            # v["_value"] looks like {'mutable_layer_1': {'layer_choice': ...} ...}
            values = v["_value"]
            for layer_name, layer_data in values.items():
                # there should be at most layer_choice, optional_inputs, optional_input_size in layer_data

                # add "_mutable_layer" as prefix so that they can be recovered later
                layer_key = _construct_general_key(k, layer_name)

                if layer_data.get("layer_choice"):  # filter out empty choice and no choice
                    layer_choice = layer_data["layer_choice"]
                else:
                    raise ValueError("No layer choice found in %s" % layer_key)

                if layer_data.get("optional_input_size"):
                    input_size = layer_data["optional_input_size"]
                    if isinstance(input_size, int):
                        input_size = [input_size, input_size]
                    if input_size[0] > input_size[1] or input_size[0] < 0:
                        _logger.error("Might not be able to handle optional_input_size < 0, please double check")
                    input_size[1] += 1
                else:
                    _logger.info("Optional input choices are set to empty by default in %s", layer_key)
                    input_size = [0, 1]

                if layer_data.get("optional_inputs"):
                    total_state_size = len(layer_data["optional_inputs"]) ** (input_size[1] - 1)
                else:
                    _logger.info("Optional inputs not found in %s", layer_key)
                    total_state_size = 1

                converted = {
                    layer_key + "/layer_choice": {
                        "_type": "choice", "_value": layer_choice
                    },
                    layer_key + "/optional_input_size": {
                        "_type": "randint", "_value": input_size
                    },
                    layer_key + "/optional_input_chosen_state": {
                        "_type": "randint", "_value": [0, total_state_size]
                    }
                }
                _logger.info(converted)
                ret.update(converted)

    return ret


def rewrite_nas_space(func):
    @functools.wraps(func)
    def wrap(self, search_space):
        search_space = convert_nas_search_space(search_space)
        return func(self, search_space)
    return wrap
