import tensorflow as tf
import tensorflow.keras as K  # type: ignore

class Identity(K.Model):
    def call(self, x):
        return x

class Concat(K.Model):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, *inputs):
        return tf.concat(inputs, self.dim)

class Split(K.Model):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, x):
        return tf.split(x, x.shape[self.dim], self.dim)

class Sum(K.Model):
    def call(self, *inputs):
        if not inputs:
            return tf.constant([[0.0]])  # FIXME: hard-coded shape for WANN
        return tf.math.add_n(inputs)

class Replication(K.Model):
    def call(self, x):
        return x, x


class Hierarchical__none(K.Model):
    def call(self, x):
        return tf.zeros([x.shape[:-1], 0], x.dtype)

class Hierarchical__1x1_conv(K.Model):
    def __init__(self):
        super().__init__()
        self.op = K.layers.Conv2D(filters=16, kernel_size=1, padding="same")
        self.batch_norm = K.layers.BatchNormalization()

    def call(self, x):
        x = tf.concat(inputs, axis=3)
        x = self.op(x)
        x = self.batch_norm(x)
        x = K.activations.relu(x)
        return x

class Hierarchical__3x3_depth_conv(K.Model):
    def __init__(self):
        super().__init__()
        self.op = K.layers.DepthwiseConv2D(kernel_size=3, padding="same")
        self.batch_norm = K.layers.BatchNormalization()

    def call(self, x):
        x = tf.concat(inputs, axis=3)
        x = self.op(x)
        x = self.batch_norm(x)
        x = K.activations.relu(x)
        padding = (inputs.shape[1] - x.shape[1], inputs.shape[2] - x.shape[2])
        return tf.pad(x, tf.constant([[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]]))

class Hierarchical__3x3_sep_conv(K.Model):
    def __init__(self):
        super().__init__()
        self.op = K.layers.SeparableConv2D(filters=16, kernel_size=3, padding="same")
        self.batch_norm = K.layers.BatchNormalization()

    def call(self, x):
        x = tf.concat(inputs, axis=3)
        x = self.op(x)
        x = self.batch_norm(x)
        x = K.activations.relu(x)
        padding = (inputs.shape[1] - x.shape[1], inputs.shape[2] - x.shape[2])
        return tf.pad(x, tf.constant([[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]]))

class Hierarchical__3x3_max_pool(K.Model):
    def __init__(self):
        super().__init__()
        self.op = K.layers.MaxPool2D(pool_size=3, padding="same")

    def call(self, x):
        x = tf.concat(inputs, axis=3)
        x = self.op(x)
        padding = (inputs.shape[1] - x.shape[1], inputs.shape[2] - x.shape[2])
        return tf.pad(x, tf.constant([[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]]))

class Hierarchical__3x3_avg_pool(K.Model):
    def __init__(self):
        super().__init__()
        self.op = K.layers.AveragePooling2D(pool_size=3, padding="same")

    def call(self, x):
        x = tf.concat(inputs, axis=3)
        x = self.op(x)
        padding = (inputs.shape[1] - x.shape[1], inputs.shape[2] - x.shape[2])
        return tf.pad(x, tf.constant([[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]]))


class Wann__activation(K.Model):
    def __init__(self, activation):
        super.__init__()
        self.activ = K.layers.Activation(activation)

    def call(self, *inputs):
        s = tf.math.add_n(*inputs)
        return self.activ(s)

class Wann__zero(K.Model):
    def call(self, *inputs):
        return tf.zeros_like(inputs[0])

class PathLevel__split(K.Model):
    def call(self, x):
        s = x.shape[3] // 2
        return x[:, :, :, :s], x[:, :, :, s : s + s]

class PathLevel__avg_pool(K.Model):
    def __init__(self):
        super().__init__()
        self.pool = K.layers.AveragePooling2D(pool_size=3, padding="same")

    def call(self, x):
        y = self.pool(x)
        padding = (x.shape[1] - y.shape[1], x.shape[2] - y.shape[2])
        return tf.pad(y, tf.constant([[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]]))


class PathLevel__max_pool(K.Model):
    def __init__(self):
        super().__init__()
        self.pool = K.layers.MaxPool2D(pool_size=3, padding="same")

    def call(self, x):
        y = self.pool(x)
        padding = (x.shape[1] - y.shape[1], x.shape[2] - y.shape[2])
        return tf.pad(y, tf.constant([[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]]))
