import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmarks.utils import json_dumps
from nni.nas.benchmarks.constants import DATABASE_DIR

# DATABASE_DIR = "/mnt/c/Users/v-ayanmao/repo/nni/nni/nas/benchmarks/nlp"
print("@DATABASE_DIR : ", DATABASE_DIR)
db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nlp.db'), autoconnect=True)
print("DATABASE_DIR : ", DATABASE_DIR)

class NlpTrialConfig(Model):
    """
    Trial config for NLP. epoch_num is fixed at 50.

    Attributes
    ----------
    arch: dict
        aka recepie in NAS-NLP-Benchmark repo
    dataset: str
        Dataset used. Could be ``PTB`` or ``WikiText-2``.
    """
    arch = JSONField(json_dumps=json_dumps, index=True)
    dataset = CharField(max_length=15, index=True, choices=[
        'ptb',
        'wikitext-2'
    ])

    class Meta:
        database = db

class NlpTrialStats(Model):
    """
    Computation statistics for NAS-NLP-Benchmark. 
    Each corresponds to one trial result after 50 epoch.

    Attributes
    ----------
    config : NlpTrialConfig
        Corresponding config for trial.
    train_loss : float or None
        Final loss on training data. Could be NaN (None).
    val_loss : float or None
        Final loss on validation data. Could be NaN (None).
    test_loss : float or None
        Final loss on test data. Could be NaN (None).
    training_time : float
        Time elapsed in seconds. aka wall_time in in NAS-NLP-Benchmark repo.
    """

    config = ForeignKeyField(NlpTrialConfig, backref='trial_stats', index=True)
    train_loss = FloatField(null=True)
    val_loss = FloatField(null=True)
    test_loss = FloatField(null=True)
    training_time = FloatField(null=True)

    class Meta:
        database = db

class NlpIntermediateStats(Model):
    """
    Computation statistics for NAS-NLP-Benchmark. 
    Each corresponds to one trial result after 50 epoch.

    Attributes
    ----------
    config : NlpTrialConfig
        Corresponding config for trial.
    train_loss : float or None
        Final loss on training data. Could be NaN (None).
    val_loss : float or None
        Final loss on validation data. Could be NaN (None).
    test_loss : float or None
        Final loss on test data. Could be NaN (None).
    training_time : float
        Time elapsed in seconds. aka wall_time in in NAS-NLP-Benchmark repo.
    
    """
    trial = ForeignKeyField(NlpTrialStats, backref='intermediates', index=True)
    current_epoch = IntegerField(index=True)
    train_loss = FloatField(null=True)
    val_loss = FloatField(null=True)
    test_loss = FloatField(null=True)
    training_time = FloatField(null=True)

    class Meta:
        database = db
    