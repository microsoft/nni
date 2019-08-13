
try:
    import tensorflow as tf
    from nni.compressors.nnimc import TfPruner
    class TfLevelPruner(TfPruner):
        def __init__(self, sparsity = 0, layer_sparsity = {}):
            super().__init__()
            self.default_sparsity = sparsity
            self.layer_sparsity = layer_sparsity

        def calc_mask(self, layer_info, weight):
            sparsity = self.layer_sparsity.get(layer_info.name, self.default_sparsity)
            threshold = tf.contrib.distributions.percentile(weight.abs(), sparsity * 100)
            return tf.cast(tf.math.greater(weight.abs(), threshold), weight.dtype)
    
    class TfAGPruner(TfPruner):
        def __init__(self, initial_sparsity=0, final_sparsity=0.8, start_epoch=1, end_epoch=1, frequency=1):
            super().__init__()
            self.initial_sparsity = initial_sparsity
            self.final_sparsity = final_sparsity
            self.start_epoch = start_epoch
            self.end_epoch = end_epoch
            self.freq = frequency
            self.now_epoch = tf.Variable(0)
            self.assign_handler = []
        
        def compute_target_sparsity(self):
            
            if self.end_epoch <= self.start_epoch :
                return self.final_sparsity
            
            now_epoch = tf.minimum(self.now_epoch, tf.constant(self.end_epoch))
            span = int(((self.end_epoch - self.start_epoch-1)//self.freq)*self.freq)
            assert span>0
            base = tf.cast(now_epoch - self.initial_sparsity, tf.float32) / span
            target_sparsity = (self.final_sparsity + 
                                (self.initial_sparsity - self.final_sparsity)*
                                (tf.pow(1.0 - base,3)))
            return target_sparsity
            
        def calc_mask(self, layer, weight):
            
            target_sparsity = self.compute_target_sparsity()
            threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
            mask = tf.stop_gradient(tf.cast(tf.math.greater(weight, threshold), weight.dtype))
            self.assign_handler.append(tf.assign(weight, weight*mask))
            return mask
            
        def update_epoch(self, sess, epoch):
            sess.run(self.assign_handler)
            sess.run(tf.assign(self.now_epoch, int(epoch)))
        
    
    class TfSensitivityPruner(TfPruner):
        def __init__(self, sparsity):
            super().__init__()
            self.sparsity = sparsity
            self.layer_mask = {}
            self.assign_handler = []

        def calc_mask(self, layer_info, weight):
            target_sparsity = self.sparsity * tf.math.reduce_std(weight) 
            mask = tf.get_variable(layer_info.name+'_mask',initializer=tf.ones(weight.shape), trainable=False)
            self.layer_mask[layer_info.name] = mask
            
            weight_assign_handler = tf.assign(weight, mask*weight)
            with tf.control_dependencies([weight_assign_handler]):
                threshold = tf.contrib.distributions.percentile(weight, target_sparsity * 100)
                new_mask = tf.stop_gradient(tf.cast(tf.math.greater(weight, threshold), weight.dtype))
                mask_update_handler = tf.assign(mask, new_mask)
                self.assign_handler.append(mask_update_handler)
            return mask

        def update_graph(self, sess):
            sess.run(self.assign_handler)

except ModuleNotFoundError:
    pass
