import tensorflow.keras as K

class Identity(K.Model):
    def call(self, x):
        return x
