import tensorflow as tf

import sdk


def load_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


class ReportMetric(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        sdk.report_intermediate_result(logs['val_accuracy'])


def train(ModelClass):
    model = ModelClass()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    (x_train, y_train), (x_test, y_test) = load_dataset()

    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=1,
        verbose=0,
        callbacks=[ReportMetric()],
        validation_data=(x_test, y_test)
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    sdk.report_final_result(accuracy)


def train_naive(ModelClass):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=1),
        ModelClass(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])
    train(lambda: model)
