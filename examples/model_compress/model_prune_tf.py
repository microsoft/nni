import argparse

import tensorflow as tf

import nni.compression.tensorflow

prune_config = {
    'level': {
        'dataset_name': 'mnist',
        'model_name': 'naive',
        'pruner_class': nni.compression.tensorflow.LevelPruner,
        'config_list': [{
            'sparsity': 0.9,
            'op_types': ['default'],
        }]
    },
}


def get_dataset(dataset_name='mnist'):
    assert dataset_name == 'mnist'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., tf.newaxis] / 255.0
    x_test = x_test[..., tf.newaxis] / 255.0
    return (x_train, y_train), (x_test, y_test)


def create_model(model_name='naive'):
    assert model_name == 'naive'
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=20, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=20, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=500),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(units=10),
        tf.keras.layers.Softmax()
    ])


def create_pruner(model, pruner_name):
    pruner_class = prune_config[pruner_name]['pruner_class']
    config_list = prune_config[pruner_name]['config_list']
    return pruner_class(model, config_list)


def main(args):
    model_name = prune_config[args.pruner_name]['model_name']
    dataset_name = prune_config[args.pruner_name]['dataset_name']
    train_set, test_set = get_dataset(dataset_name)
    model = create_model(model_name)

    print('start training')
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_set[0],
        train_set[1],
        batch_size=args.batch_size,
        epochs=args.pretrain_epochs,
        validation_data=test_set
    )

    print('start model pruning')
    optimizer_finetune = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=1e-4)
    pruner = create_pruner(model, args.pruner_name)
    model = pruner.compress()
    model.compile(
        optimizer=optimizer_finetune,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True  # NOTE: Important, model compression does not work in graph mode!
    )
    model.fit(
        train_set[0],
        train_set[1],
        batch_size=args.batch_size,
        epochs=args.prune_epochs,
        validation_data=test_set
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruner_name', type=str, default='level')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--prune_epochs', type=int, default=10)

    args = parser.parse_args()
    main(args)
