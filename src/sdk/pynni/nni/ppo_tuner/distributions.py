import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

from util import fc

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x):
        return - self.neglogp(x)
    def get_shape(self):
        return self.flatparam().shape
    @property
    def shape(self):
        return self.get_shape()
    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        raise NotImplementedError
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = _matching_fc(latent_vector, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec.astype('int32')
        assert (self.ncats > 0).all()
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def pdfromlatent(self, latent, init_scale=1.0, init_bias=0.0):
        pdparam = _matching_fc(latent, 'pi', self.ncats.sum(), init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return tf.int32

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        mean = _matching_fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        return self.pdfromflat(pdparam), mean

    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return BernoulliPd
    def param_shape(self):
        return [self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.int32
    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = _matching_fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)
    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=x)
    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = list(map(CategoricalPd,
            tf.split(flat, np.array(nvec, dtype=np.int32), axis=-1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])
    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)
    def flatparam(self):
        return self.logits
    @property
    def mean(self):
        return self.ps
    def mode(self):
        return tf.round(self.ps)
    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=-1)
    def kl(self, other):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=-1) - tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)
    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.ps))
        return tf.to_float(math_ops.less(u, self.ps))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError


def _matching_fc(tensor, name, size, init_scale, init_bias):
    if tensor.shape[-1] == size:
        return tensor
    else:
        return fc(tensor, name, size, init_scale=init_scale, init_bias=init_bias)