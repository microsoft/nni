"""
To migrate the old chinese translation into new ones.
"""

from pathlib import Path

whitelist = [
    'Overview.rst',
    'installation.rst',
    'Tutorial/InstallationLinux.rst',
    'Tutorial/InstallationWin.rst',
    'Tutorial/QuickStart.rst',
    'TrialExample/Trials.rst',
    'Tutorial/WebUI.rst',
    'NAS/QuickStart.rst',
    'Compression/Overview.rst',
    'Compression/QuickStart.rst',

    'model_compression.rst',
    'Compression/quantization.rst',
    'Compression/advanced.rst',
    'Compression/pruning.rst',
    'Compression/v2_pruning.rst',
    'hyperparameter_tune.rst',
    'NAS/construct_space.rst',
    'NAS/multi_trial_nas.rst',
    'NAS/one_shot_nas.rst',
    'examples.rst',
    'reference.rst',
    'installation.rst',
    'CommunitySharings/model_compression.rst',
    'CommunitySharings/autosys.rst',
    'CommunitySharings/automodel.rst',
    'CommunitySharings/perf_compare.rst',
    'CommunitySharings/feature_engineering.rst',
    'training_services.rst',
    'contribution.rst',
    'builtin_tuner.rst',
    'nas.rst',
    'hpo_advanced.rst',
    'feature_engineering.rst',
    'builtin_assessor.rst',
    'sdk_reference.rst'
]

translation_strings = {}

import shutil

for i, path in enumerate(whitelist):
    shutil.copyfile(Path('zh_CN') / path, Path('en_US') / path.replace('.rst', '_zh_CN.rst'))
