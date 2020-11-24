import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmarks.constants import DATABASE_DIR
from nni.nas.benchmarks.utils import json_dumps

db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nds.db'), autoconnect=True)


class NdsTrialConfig(Model):
    """
    Trial config for NDS.

    Attributes
    ----------
    model_family : str
        Could be ``nas_cell``, ``residual_bottleneck``, ``residual_basic`` or ``vanilla``.
    model_spec : dict
        If ``model_family`` is ``nas_cell``, it contains ``num_nodes_normal``, ``num_nodes_reduce``, ``depth``,
        ``width``, ``aux`` and ``drop_prob``. If ``model_family`` is ``residual_bottleneck``, it contains ``bot_muls``,
        ``ds`` (depths), ``num_gs`` (number of groups) and ``ss`` (strides). If ``model_family`` is ``residual_basic`` or
        ``vanilla``, it contains ``ds``, ``ss`` and ``ws``.
    cell_spec : dict
        If ``model_family`` is not ``nas_cell`` it will be an empty dict. Otherwise, it specifies
        ``<normal/reduce>_<i>_<op/input>_<x/y>``, where i ranges from 0 to ``num_nodes_<normal/reduce> - 1``.
        If it is an ``op``, the value is chosen from the constants specified previously like :const:`nni.nas.benchmark.nds.CONV_1X1`.
        If it is i's ``input``, the value range from 0 to ``i + 1``, as ``nas_cell`` uses previous two nodes as inputs, and
        node 0 is actually the second node. Refer to NASNet paper for details. Finally, another two key-value pairs
        ``normal_concat`` and ``reduce_concat`` specify which nodes are eventually concatenated into output.
    dataset : str
        Dataset used. Could be ``cifar10`` or ``imagenet``.
    generator : str
        Can be one of ``random`` which generates configurations at random, while keeping learning rate and weight decay fixed,
        ``fix_w_d`` which further keeps ``width`` and ``depth`` fixed, only applicable for ``nas_cell``. ``tune_lr_wd`` which
        further tunes learning rate and weight decay.
    proposer : str
        Paper who has proposed the distribution for random sampling. Available proposers include ``nasnet``, ``darts``, ``enas``,
        ``pnas``, ``amoeba``, ``vanilla``, ``resnext-a``, ``resnext-b``, ``resnet``, ``resnet-b`` (ResNet with bottleneck).
        See NDS paper for details.
    base_lr : float
        Initial learning rate.
    weight_decay : float
        L2 weight decay applied on weights.
    num_epochs : int
        Number of epochs scheduled, during which learning rate will decay to 0 following cosine annealing.
    """

    model_family = CharField(max_length=20, index=True, choices=[
        'nas_cell',
        'residual_bottleneck',
        'residual_basic',
        'vanilla',
    ])
    model_spec = JSONField(json_dumps=json_dumps, index=True)
    cell_spec = JSONField(json_dumps=json_dumps, index=True, null=True)
    dataset = CharField(max_length=15, index=True, choices=['cifar10', 'imagenet'])
    generator = CharField(max_length=15, index=True, choices=[
        'random',
        'fix_w_d',
        'tune_lr_wd',
    ])
    proposer = CharField(max_length=15, index=True)
    base_lr = FloatField()
    weight_decay = FloatField()
    num_epochs = IntegerField()

    class Meta:
        database = db


class NdsTrialStats(Model):
    """
    Computation statistics for NDS. Each corresponds to one trial.

    Attributes
    ----------
    config : NdsTrialConfig
        Corresponding config for trial.
    seed : int
        Random seed selected, for reproduction.
    final_train_acc : float
        Final accuracy on training data, ranging from 0 to 100.
    final_train_loss : float or None
        Final cross entropy loss on training data. Could be NaN (None).
    final_test_acc : float
        Final accuracy on test data, ranging from 0 to 100.
    best_train_acc : float
        Best accuracy on training data, ranging from 0 to 100.
    best_train_loss : float or None
        Best cross entropy loss on training data. Could be NaN (None).
    best_test_acc : float
        Best accuracy on test data, ranging from 0 to 100.
    parameters : float
        Number of trainable parameters in million.
    flops : float
        FLOPs in million.
    iter_time : float
        Seconds elapsed for each iteration.
    """
    config = ForeignKeyField(NdsTrialConfig, backref='trial_stats', index=True)
    seed = IntegerField()
    final_train_acc = FloatField()
    final_train_loss = FloatField(null=True)
    final_test_acc = FloatField()
    best_train_acc = FloatField()
    best_train_loss = FloatField(null=True)
    best_test_acc = FloatField()
    parameters = FloatField()
    flops = FloatField()
    iter_time = FloatField()

    class Meta:
        database = db


class NdsIntermediateStats(Model):
    """
    Intermediate statistics for NDS.

    Attributes
    ----------
    trial : NdsTrialStats
        Corresponding trial.
    current_epoch : int
        Elapsed epochs.
    train_loss : float or None
        Current cross entropy loss on training data. Can be NaN (None).
    train_acc : float
        Current accuracy on training data, ranging from 0 to 100.
    test_acc : float
        Current accuracy on test data, ranging from 0 to 100.
    """

    trial = ForeignKeyField(NdsTrialStats, backref='intermediates', index=True)
    current_epoch = IntegerField(index=True)
    train_loss = FloatField(null=True)
    train_acc = FloatField()
    test_acc = FloatField()

    class Meta:
        database = db
