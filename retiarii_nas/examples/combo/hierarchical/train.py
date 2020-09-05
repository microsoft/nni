import tensorflow as tf
import sdk

def load_dataset():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

class ReportMetric(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs)
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
        validation_data=(x_test, y_test),
        steps_per_epoch=10
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    sdk.report_final_result(accuracy)
