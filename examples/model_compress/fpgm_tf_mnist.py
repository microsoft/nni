import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from nni.compression.tensorflow import FPGMPruner

def get_data():
    (X_train_full, y_train_full), _ = keras.datasets.mnist.load_data()
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + 1e-7
    X_train = (X_train - X_mean) / X_std
    X_valid = (X_valid - X_mean) / X_std

    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]

    return X_train, X_valid, y_train, y_valid

def get_model():
    model = keras.models.Sequential([
        Conv2D(filters=32, kernel_size=7, input_shape=[28, 28, 1], activation='relu', padding="SAME"),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, activation='relu', padding="SAME"),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dropout(0.5),
        Dense(units=10, activation='softmax'),
    ])
    model.compile(loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.SGD(lr=1e-3),
        metrics=["accuracy"])
    return model

def main():
    X_train, X_valid, y_train, y_valid = get_data()
    model = get_model()

    configure_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2D']
    }]
    pruner = FPGMPruner(model, configure_list)
    pruner.compress()

    update_epoch_callback = keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: pruner.update_epoch(epoch))

    model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[update_epoch_callback])


if __name__ == '__main__':
    main()
