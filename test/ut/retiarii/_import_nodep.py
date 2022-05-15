"""To test the cases of importing NAS without certain DL libraries installed."""

import argparse
import sys


def import_related():
    import nni.retiarii
    # import nni.retiarii.strategy  # FIXME: this doesn't work yet
    import nni.retiarii.experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('masked', choices=['torch', 'tensorflow'])
    args = parser.parse_args()
    if args.masked == 'torch':
        # https://stackoverflow.com/questions/1350466/preventing-python-code-from-importing-certain-modules
        sys.modules['torch'] = None
        import_related()
    elif args.masked == 'tensorflow':
        sys.modules['tensorflow'] = None
        import_related()


if __name__ == '__main__':
    main()
