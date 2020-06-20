import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmark.constants import DATABASE_DIR

db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nasbench201.db'), autoconnect=True)


class Nb201RunConfig(Model):
    """
    Run config for NAS-Bench-201.
    """

    arch = JSONField(index=True)
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


class Nb201ComputedStats(Model):
    """
    Computation statistics for NAS-Bench-201. Each corresponds to one run.

    Attributes
    ----------
    config : 

    """
    config = ForeignKeyField(Nb201RunConfig, backref='computed_stats', index=True)
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

    """

    run = ForeignKeyField(Nb201ComputedStats, backref='intermediates', index=True)
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
