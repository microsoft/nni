import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmarks.constants import DATABASE_DIR
from nni.nas.benchmarks.utils import json_dumps

db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nasbench201.db'), autoconnect=True)


class Nb201TrialConfig(Model):
    """
    Trial config for NAS-Bench-201.

    Attributes
    ----------
    arch : dict
        A dict with keys ``0_1``, ``0_2``, ``0_3``, ``1_2``, ``1_3``, ``2_3``, each of which
        is an operator chosen from :const:`nni.nas.benchmark.nasbench201.NONE`,
        :const:`nni.nas.benchmark.nasbench201.SKIP_CONNECT`,
        :const:`nni.nas.benchmark.nasbench201.CONV_1X1`,
        :const:`nni.nas.benchmark.nasbench201.CONV_3X3` and :const:`nni.nas.benchmark.nasbench201.AVG_POOL_3X3`.
    num_epochs : int
        Number of epochs planned for this trial. Should be one of 12 and 200.
    num_channels: int
        Number of channels for initial convolution. 16 by default.
    num_cells: int
        Number of cells per stage. 5 by default.
    dataset: str
        Dataset used for training and evaluation. NAS-Bench-201 provides the following 4 options:
        ``cifar10-valid`` (training data is splited into 25k for training and 25k for validation,
        validation data is used for test), ``cifar10`` (training data is used in training, validation
        data is splited into 5k for validation and 5k for testing), ``cifar100`` (same protocol as ``cifar10``),
        and ``imagenet16-120`` (a subset of 120 classes in ImageNet, downscaled to 16x16, using training data
        for training, 6k images from validation set for validation and the other 6k for testing).
    """

    arch = JSONField(json_dumps=json_dumps, index=True)
    num_epochs = IntegerField(index=True)
    num_channels = IntegerField()
    num_cells = IntegerField()
    dataset = CharField(max_length=20, index=True, choices=[
        'cifar10-valid',  # 25k+25k+10k
        'cifar10',  # 50k+5k+5k
        'cifar100',  # 50k+5k+5k
        'imagenet16-120',
    ])

    class Meta:
        database = db


class Nb201TrialStats(Model):
    """
    Computation statistics for NAS-Bench-201. Each corresponds to one trial.

    Attributes
    ----------
    config : Nb201TrialConfig
        Setup for this trial data.
    seed : int
        Random seed selected, for reproduction.
    train_acc : float
        Final accuracy on training data, ranging from 0 to 100.
    valid_acc : float
        Final accuracy on validation data, ranging from 0 to 100.
    test_acc : float
        Final accuracy on test data, ranging from 0 to 100.
    ori_test_acc : float
        Test accuracy on original validation set (10k for CIFAR and 12k for Imagenet16-120),
        ranging from 0 to 100.
    train_loss : float or None
        Final cross entropy loss on training data. Note that loss could be NaN, in which case
        this attributed will be None.
    valid_loss : float or None
        Final cross entropy loss on validation data.
    test_loss : float or None
        Final cross entropy loss on test data.
    ori_test_loss : float or None
        Final cross entropy loss on original validation set.
    parameters : float
        Number of trainable parameters in million.
    latency : float
        Latency in seconds.
    flops : float
        FLOPs in million.
    training_time : float
        Duration of training in seconds.
    valid_evaluation_time : float
        Time elapsed to evaluate on validation set.
    test_evaluation_time : float
        Time elapsed to evaluate on test set.
    ori_test_evaluation_time : float
        Time elapsed to evaluate on original test set.
    """
    config = ForeignKeyField(Nb201TrialConfig, backref='trial_stats', index=True)
    seed = IntegerField()
    train_acc = FloatField()
    valid_acc = FloatField()
    test_acc = FloatField()
    ori_test_acc = FloatField()  # test accuracy of the original test set
    train_loss = FloatField(null=True)  # possibly nan
    valid_loss = FloatField(null=True)
    test_loss = FloatField(null=True)
    ori_test_loss = FloatField(null=True)
    parameters = FloatField()  # parameters in million
    latency = FloatField()  # latency in milliseconds
    flops = FloatField()  # flops in million
    training_time = FloatField()
    valid_evaluation_time = FloatField()
    test_evaluation_time = FloatField()
    ori_test_evaluation_time = FloatField()

    class Meta:
        database = db


class Nb201IntermediateStats(Model):
    """
    Intermediate statistics for NAS-Bench-201.

    Attributes
    ----------
    trial : Nb201TrialStats
        Corresponding trial.
    current_epoch : int
        Elapsed epochs.
    train_acc : float
        Current accuracy on training data, ranging from 0 to 100.
    valid_acc : float
        Current accuracy on validation data, ranging from 0 to 100.
    test_acc : float
        Current accuracy on test data, ranging from 0 to 100.
    ori_test_acc : float
        Test accuracy on original validation set (10k for CIFAR and 12k for Imagenet16-120),
        ranging from 0 to 100.
    train_loss : float or None
        Current cross entropy loss on training data.
    valid_loss : float or None
        Current cross entropy loss on validation data.
    test_loss : float or None
        Current cross entropy loss on test data.
    ori_test_loss : float or None
        Current cross entropy loss on original validation set.
    """

    trial = ForeignKeyField(Nb201TrialStats, backref='intermediates', index=True)
    current_epoch = IntegerField(index=True)
    train_acc = FloatField(null=True)
    valid_acc = FloatField(null=True)
    test_acc = FloatField(null=True)
    ori_test_acc = FloatField(null=True)
    train_loss = FloatField(null=True)
    valid_loss = FloatField(null=True)
    test_loss = FloatField(null=True)
    ori_test_loss = FloatField(null=True)

    class Meta:
        database = db
