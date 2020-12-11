"""
Entrypoint for trials.

Assuming execution engine is BaseExecutionEngine.
"""
import os

from .execution.base import BaseExecutionEngine
from .execution.cgo_engine import CGOExecutionEngine

if __name__ == '__main__':
    if os.environ.get('CGO') == 'true':
        CGOExecutionEngine.trial_execute_graph()
    else:
        BaseExecutionEngine.trial_execute_graph()
