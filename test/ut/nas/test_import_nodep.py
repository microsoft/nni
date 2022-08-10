"""To test the cases of importing NAS without certain DL libraries installed."""

import argparse
import subprocess
import sys

import pytest


def import_related(mask_out):
    import nni
    nni.set_default_framework(mask_out)
    import nni.retiarii
    import nni.retiarii.evaluator
    import nni.retiarii.hub
    import nni.retiarii.strategy  # FIXME: this doesn't work yet
    import nni.retiarii.experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('masked', choices=['torch', 'torch_none', 'tensorflow'])
    args = parser.parse_args()
    if args.masked == 'torch':
        # https://stackoverflow.com/questions/1350466/preventing-python-code-from-importing-certain-modules
        sys.modules['torch'] = None
        import_related('tensorflow')
    if args.masked == 'torch_none':
        sys.modules['torch'] = None
        import_related('none')
    elif args.masked == 'tensorflow':
        sys.modules['tensorflow'] = None
        import_related('pytorch')


@pytest.mark.parametrize('framework', ['torch', 'torch_none', 'tensorflow'])
def test_import_without_framework(framework):
    subprocess.run([sys.executable, __file__, framework], check=True)


if __name__ == '__main__':
    main()
