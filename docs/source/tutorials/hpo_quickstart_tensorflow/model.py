"""
Port TensorFlow Quickstart to NNI
=================================
This is a modified version of `TensorFlow quickstart`_.

It can be run directly and will have the exact same result as original version.

Furthermore, it enables the ability of auto-tuning with an NNI *experiment*, which will be discussed later.

For now, we recommend to run this script directly to verify the environment.

There are only 3 key differences from the original version:

 1. In `Get optimized hyperparameters`_ part, it receives auto-generated hyperparameters.
 2. In `(Optional) Report intermediate results`_ part, it reports per-epoch accuracy for visualization.
 3. In `Report final result`_ part, it reports final accuracy for tuner to generate next hyperparameter set.

.. _TensorFlow quickstart: https://www.tensorflow.org/tutorials/quickstart/beginner
"""

# %%
import nni
import tensorflow as tf

# %%
# Hyperparameters to be tuned
# ---------------------------
params = {
    'dense_units': 128,
    'activation_type': 'relu',
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
}

# %%
# Get optimized hyperparameters
# -----------------------------
# If run directly, ``nni.get_next_parameters()`` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

# %%
# Load dataset
# ------------
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# %%
# Build model with hyperparameters
# --------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(params['dense_units'], activation=params['activation_type']),
    tf.keras.layers.Dropout(params['dropout_rate']),
    tf.keras.layers.Dense(10)
])

adam = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=adam, loss=loss_fn, metrics=['accuracy'])

# %%
# (Optional) Report intermediate results
# --------------------------------------
# The callback reports per-epoch accuracy to show learning curve in NNI web portal.
# And in :doc:`/hpo/assessors`, you will see how to leverage the metrics for early stopping.
#
# You can safely skip this and the experiment will work fine.
callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end = lambda epoch, logs: nni.report_intermediate_result(logs['accuracy'])
)

# %%
# Train and evluate the model
# ---------------------------
model.fit(x_train, y_train, epochs=5, verbose=2, callbacks=[callback])
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

# %%
# Report final result
# -------------------
# Report final accuracy to NNI so the tuning algorithm can predict best hyperparameters.
nni.report_final_result(accuracy)
