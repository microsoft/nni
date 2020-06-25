import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmark.constants import DATABASE_DIR

db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nds.db'), autoconnect=True)


class NdsRunConfig(Model):
    """
    Run config for NDS.

    Attributes
    ----------
    model_family : str
        Could be ``nas_cell``, ``residual_bottleneck``, ``residual_basic`` or ``vanilla``.
    model_spec : dict
        TODO
    cell_spec : dict
        TODO
    dataset : str
        Dataset used. Could be ``cifar10`` or ``imagenet``.
    generator : str
        TODO
    proposer : str
        TODO
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
    model_spec = JSONField(index=True)
    cell_spec = JSONField(index=True, null=True)
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


class NdsComputedStats(Model):
    """
    Computation statistics for NDS. Each corresponds to one run.

    Attributes
    ----------
    config : NdsRunConfig
        Corresponding config for run.
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
    config = ForeignKeyField(NdsRunConfig, backref='computed_stats', index=True)
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
    run : NdsComputedStats
        Corresponding run.
    current_epoch : int
        Elapsed epochs.
    train_loss : float or None
        Current cross entropy loss on training data. Can be NaN (None).
    train_acc : float
        Current accuracy on training data, ranging from 0 to 100.
    test_acc : float
        Current accuracy on test data, ranging from 0 to 100.
    """

    run = ForeignKeyField(NdsComputedStats, backref='intermediates', index=True)
    current_epoch = IntegerField(index=True)
    train_loss = FloatField(null=True)
    train_acc = FloatField()
    test_acc = FloatField()

    class Meta:
        database = db
