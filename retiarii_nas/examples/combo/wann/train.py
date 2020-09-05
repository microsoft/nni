import tensorflow as tf
import sdk

def load_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., tf.newaxis] / 255.0
    x_test = x_test[..., tf.newaxis] / 255.0
    return (x_train, y_train), (x_test, y_test)

def train(ModelClass):
    model = ModelClass()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    (x_train, y_train), (x_test, y_test) = load_dataset()

    # WANN does not train

    print('## y_test:', y_test)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    sdk.report_final_result(accuracy)
