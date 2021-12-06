# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from nni.tools.package_utils import get_registered_algo_meta

def test_old():
    with TemporaryDirectory() as tmp_dir:
        config_dir = Path(tmp_dir, 'nni')
        os.environ['NNI_CONFIG_DIR'] = str(config_dir)

        # mock upgrading from old version
        config_dir.mkdir()
        asset_config = Path(__file__).parent / 'assets' / 'registered_algorithms.yml'
        shutil.copyfile(asset_config, config_dir / 'registered_algorithms.yml')

        tpe = get_registered_algo_meta('TPE')
        assert tpe['className'] == 'nni.algorithms.hpo.tpe_tuner.TpeTuner'

        custom = get_registered_algo_meta('demotuner')
        assert custom['className'] == 'demo_tuner.DemoTuner'

        os.environ.pop('NNI_CONFIG_DIR')

def test_new():
    with TemporaryDirectory() as tmp_dir:
        config_dir = Path(tmp_dir, 'nni')
        os.environ['NNI_CONFIG_DIR'] = str(config_dir)

        # mock fresh install

        for _ in range(2):
            tpe = get_registered_algo_meta('TPE')
            assert tpe['className'] == 'nni.algorithms.hpo.tpe_tuner.TpeTuner'

        os.environ.pop('NNI_CONFIG_DIR')

if __name__ == '__main__':
    test_old()
    test_new()
