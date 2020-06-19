import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmark.constants import DATABASE_DIR

db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nasbench101.db'), autoconnect=True)


class Nb101RunConfig(Model):
    """
    Run config for NAS-Bench-101.
    """

    arch = JSONField(index=True)
    num_vertices = IntegerField(index=True)
    hash = CharField(max_length=64, index=True)
    num_epochs = IntegerField(index=True)

    class Meta:
        database = db


class Nb101ComputedStats(Model):
    """
    Computation statistics for NAS-Bench-101. Each corresponds to one run.
    Each config has multiple runs. Unfortunately, seed for each run is not available.

    Attributes
    ----------
    config : 

    """
    config = ForeignKeyField(Nb101RunConfig, backref='computed_stats')
    train_acc = FloatField()
    valid_acc = FloatField()
    test_acc = FloatField()
    parameters = FloatField()
    training_time = FloatField()

    class Meta:
        database = db


class Nb101IntermediateStats(Model):
    """
    Intermediate statistics for NAS-Bench-101.

    """

    run = ForeignKeyField(Nb101ComputedStats, backref='intermediates')
    current_epoch = IntegerField(index=True)
    train_acc = FloatField()
    valid_acc = FloatField()
    test_acc = FloatField()
    training_time = FloatField()

    class Meta:
        database = db
