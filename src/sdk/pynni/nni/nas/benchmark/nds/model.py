import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmark.constants import DATABASE_DIR

db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nds.db'), autoconnect=True)


class NdsRunConfig(Model):
    """
    Run config for NDS.
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
    config : 

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

    """

    run = ForeignKeyField(NdsComputedStats, backref='intermediates', index=True)
    current_epoch = IntegerField(index=True)
    train_loss = FloatField(null=True)
    train_acc = FloatField()
    test_acc = FloatField()

    class Meta:
        database = db
