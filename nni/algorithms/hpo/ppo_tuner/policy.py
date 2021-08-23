# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
build policy/value network from model
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .distri import CategoricalPdType
from .util import lstm_model, fc, observation_placeholder, adjust_shape


class PolicyWithValue:
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, np_mask=None, is_act_model=False, **tensors):
        """
        Parameters
        ----------
        env : obj
            RL environment
        observations : tensorflow placeholder
            Tensorflow placeholder in which the observations will be fed
        latent : tensor
            Latent state from which policy distribution parameters should be inferred
        vf_latent : tensor
            Latent state from which value function should be inferred (if None, then latent is used)
        sess : tensorflow session
            Tensorflow session to run calculations in (if None, default session is used)
        **tensors
            Tensorflow tensors for additional attributes such as state or mask
        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.np_mask = np_mask
        self.pdtype = CategoricalPdType(env.action_space.n, env.nsteps, np_mask, is_act_model)

        self.act_latent = latent
        self.nh = env.action_space.n

        self.pd, self.pi, self.mask, self.mask_npinf = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        assert estimate_q is False
        self.vf = fc(vf_latent, 'vf', 1)
        self.vf = self.vf[:, 0]

        if is_act_model:
            self._build_model_for_step()

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def _build_model_for_step(self):
        # multiply with weight and apply mask on self.act_latent to generate
        self.act_step = step = tf.placeholder(shape=(), dtype=tf.int64, name='act_step')
        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            from .util import ortho_init
            nin = self.act_latent.get_shape()[1].value
            w = tf.get_variable("w", [nin, self.nh], initializer=ortho_init(0.01))
            b = tf.get_variable("b", [self.nh], initializer=tf.constant_initializer(0.0))
            logits = tf.matmul(self.act_latent, w)+b
            piece = tf.slice(self.mask, [step, 0], [1, self.nh])
            re_piece = tf.reshape(piece, [-1])
            masked_logits = tf.math.multiply(logits, re_piece)

            npinf_piece = tf.slice(self.mask_npinf, [step, 0], [1, self.nh])
            re_npinf_piece = tf.reshape(npinf_piece, [-1])

        def sample(logits, mask_npinf):
            new_logits = tf.math.add(logits, mask_npinf)
            u = tf.random_uniform(tf.shape(new_logits), dtype=logits.dtype)
            return tf.argmax(new_logits - tf.log(-1*tf.log(u)), axis=-1)

        def neglogp(logits, x):
            # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
            # Note: we can't use sparse_softmax_cross_entropy_with_logits because
            #       the implementation does not allow second-order derivatives...
            if x.dtype in {tf.uint8, tf.int32, tf.int64}:
                # one-hot encoding
                x_shape_list = x.shape.as_list()
                logits_shape_list = logits.get_shape().as_list()[:-1]
                for xs, ls in zip(x_shape_list, logits_shape_list):
                    if xs is not None and ls is not None:
                        assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

                x = tf.one_hot(x, logits.get_shape().as_list()[-1])
            else:
                # already encoded
                assert x.shape.as_list() == logits.shape.as_list()

            return tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=x)

        self.act_action = sample(masked_logits, re_npinf_piece)
        self.act_neglogp = neglogp(masked_logits, self.act_action)


    def step(self, step, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters
        ----------
        observation : np array
            Observation data (either single or a batch)
        **extra_feed
            Additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        extra_feed['act_step'] = step
        a, v, state, neglogp = self._evaluate([self.act_action, self.vf, self.state, self.act_neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters
        ----------
        observation : np array
            Observation data (either single or a batch)
        **extra_feed
            Additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns
        -------
        Value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)


def build_lstm_policy(model_config, value_network=None, estimate_q=False, **policy_kwargs):
    """
    Build lstm policy and value network, they share the same lstm network.
    the parameters all use their default values.

    Parameter
    ---------
    model_config : obj
        Configurations of the model
    value_network : obj
        The network for value function
    estimate_q : bool
        Whether to estimate ``q``
    **policy_kwargs
        The kwargs for policy network, i.e., lstm model

    Returns
    -------
    func
        The policy network
    """
    policy_network = lstm_model(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, np_mask=None, is_act_model=False):
        ob_space = model_config.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        # encode_observation is not necessary anymore as we use embedding_lookup
        encoded_x = X

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x, 1, model_config.observation_space.n)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv, model_config.observation_space.n)
                    extra_tensors.update(recurrent_tensors)

        _v_net = value_network

        assert _v_net is None or _v_net == 'shared'
        vf_latent = policy_latent

        policy = PolicyWithValue(
            env=model_config,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            np_mask=np_mask,
            is_act_model=is_act_model,
            **extra_tensors
        )
        return policy

    return policy_fn
