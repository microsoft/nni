# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entrypoint for trials.

Assuming execution engine is BaseExecutionEngine.
"""
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exec', choices=['base', 'py', 'cgo'])
    args = parser.parse_args()
    if args.exec == 'base':
        from .execution.base import BaseExecutionEngine
        engine = BaseExecutionEngine
    elif args.exec == 'cgo':
        from .execution.cgo_engine import CGOExecutionEngine
        engine = CGOExecutionEngine
    elif args.exec == 'py':
        from .execution.python import PurePythonExecutionEngine
        engine = PurePythonExecutionEngine
    engine.trial_execute_graph()
