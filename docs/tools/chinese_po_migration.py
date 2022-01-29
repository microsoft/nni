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

for i, path in enumerate(whitelist):
    english_content = (Path('en_US') / path).read_text().strip().splitlines()
    chinese_content = (Path('zh_CN') / path).read_text().strip().splitlines()[2:]

    # print(path)
    # for en, ch in zip(english_content, chinese_content):
    #     en = en.strip()
    #     ch = ch.strip()
    #     if not en:
    #         continue
    #     if en == ch:
    #         continue
    #     if all(s == '#' for s in en):
    #         continue
    #     if ':maxdepth:' in en:
    #         continue
    #     if en.endswith('>'):
    #         print(en[:en.find('<')].strip(), ch[:ch.find('<')].strip())
    #     else:
    #         print(en, ch)
    print(path, len(english_content), len(chinese_content))
    # if i == 5:
    #     break
