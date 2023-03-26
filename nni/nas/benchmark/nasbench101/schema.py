# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model, Proxy
from playhouse.sqlite_ext import JSONField

from nni.nas.benchmark.utils import json_dumps

proxy = Proxy()


class Nb101TrialConfig(Model):
    """
    Trial config for NAS-Bench-101.

    Attributes
    ----------
    arch : dict
        A dict with keys ``op1``, ``op2``, ... and ``input1``, ``input2``, ... Vertices are
        enumerate from 0. Since node 0 is input node, it is skipped in this dict. Each ``op``
        is one of :const:`nni.nas.benchmark.nasbench101.CONV3X3_BN_RELU`,
        :const:`nni.nas.benchmark.nasbench101.CONV1X1_BN_RELU`, and :const:`nni.nas.benchmark.nasbench101.MAXPOOL3X3`.
        Each ``input`` is a list of previous nodes. For example ``input5`` can be ``[0, 1, 3]``.
    num_vertices : int
        Number of vertices (nodes) in one cell. Should be less than or equal to 7 in default setup.
    hash : str
        Graph-invariant MD5 string for this architecture.
    num_epochs : int
        Number of epochs planned for this trial. Should be one of 4, 12, 36, 108 in default setup.
    """

    arch = JSONField(json_dumps=json_dumps, index=True)
    num_vertices = IntegerField(index=True)
    hash = CharField(max_length=64, index=True)
    num_epochs = IntegerField(index=True)

    class Meta:
        database = proxy


class Nb101TrialStats(Model):
    """
    Computation statistics for NAS-Bench-101. Each corresponds to one trial.
    Each config has multiple trials with different random seeds, but unfortunately seed for each trial is unavailable.
    NAS-Bench-101 trains and evaluates on CIFAR-10 by default. The original training set is divided into
    40k training images and 10k validation images, and the original validation set is used for test only.

    Attributes
    ----------
    config : Nb101TrialConfig
        Setup for this trial data.
    train_acc : float
        Final accuracy on training data, ranging from 0 to 100.
    valid_acc : float
        Final accuracy on validation data, ranging from 0 to 100.
    test_acc : float
        Final accuracy on test data, ranging from 0 to 100.
    parameters : float
        Number of trainable parameters in million.
    training_time : float
        Duration of training in seconds.
    """
    config = ForeignKeyField(Nb101TrialConfig, backref='trial_stats', index=True)
    train_acc = FloatField()
    valid_acc = FloatField()
    test_acc = FloatField()
    parameters = FloatField()
    training_time = FloatField()

    class Meta:
        database = proxy


class Nb101IntermediateStats(Model):
    """
    Intermediate statistics for NAS-Bench-101.

    Attributes
    ----------
    trial : Nb101TrialStats
        The exact trial where the intermediate result is produced.
    current_epoch : int
        Elapsed epochs when evaluation is done.
    train_acc : float
        Intermediate accuracy on training data, ranging from 0 to 100.
    valid_acc : float
        Intermediate accuracy on validation data, ranging from 0 to 100.
    test_acc : float
        Intermediate accuracy on test data, ranging from 0 to 100.
    training_time : float
        Time elapsed in seconds.
    """

    trial = ForeignKeyField(Nb101TrialStats, backref='intermediates', index=True)
    current_epoch = IntegerField(index=True)
    train_acc = FloatField()
    valid_acc = FloatField()
    test_acc = FloatField()
    training_time = FloatField()

    class Meta:
        database = proxy
