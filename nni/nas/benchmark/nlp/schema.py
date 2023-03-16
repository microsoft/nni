# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from peewee import CharField, FloatField, ForeignKeyField, IntegerField, Model
from playhouse.sqlite_ext import JSONField, SqliteExtDatabase

from nni.nas.benchmark.utils import json_dumps
from nni.nas.benchmark.constants import DATABASE_DIR

db = SqliteExtDatabase(os.path.join(DATABASE_DIR, 'nlp.db'), autoconnect=True)


class NlpTrialConfig(Model):
    """
    Trial config for NLP. epoch_num is fixed at 50.

    Attributes
    ----------
    arch: dict
        aka recepie in NAS-NLP-Benchmark repo (https://github.com/fmsnew/nas-bench-nlp-release).
        an arch has multiple Node, Node_input_n and Node_op.
        ``Node`` can be ``node_n`` or ``h_new_n`` or ``f/i/o/j(_act)`` etc. (n is an int number and need not to be consecutive)
        ``Node_input_n`` can be ``Node`` or ``x`` etc.
        ``Node_op`` can be ``linear`` or ``activation_sigm`` or ``activation_tanh`` or ``elementwise_prod``
        or ``elementwise_sum`` or ``activation_leaky_relu`` ...
        e.g., {"h_new_0_input_0":"node_3","h_new_0_input_1":"x","h_new_0_op":"linear","node_2_input_0":"x",
        "node_2_input_1":"h_prev_0","node_2_op":"linear","node_3_input_0":"node_2","node_3_op":"activation_leaky_relu"}
    dataset: str
        Dataset used. Could be ``ptb`` or ``wikitext-2``.
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
    Each corresponds to one trial result for 1-50 epoch.

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
