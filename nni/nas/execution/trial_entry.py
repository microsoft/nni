# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entrypoint for trials.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exec', choices=['base', 'py', 'cgo', 'benchmark'])
    args = parser.parse_args()
    if args.exec == 'base':
        from .pytorch.graph import BaseExecutionEngine
        engine = BaseExecutionEngine
    elif args.exec == 'cgo':
        from .pytorch.cgo import CGOExecutionEngine
        engine = CGOExecutionEngine
    elif args.exec == 'py':
        from .pytorch.simplified import PurePythonExecutionEngine
        engine = PurePythonExecutionEngine
    elif args.exec == 'benchmark':
        from .pytorch.benchmark import BenchmarkExecutionEngine
        engine = BenchmarkExecutionEngine
    else:
        raise ValueError(f'Unrecognized benchmark name: {args.exec}')
    engine.trial_execute_graph()


if __name__ == '__main__':
    main()
