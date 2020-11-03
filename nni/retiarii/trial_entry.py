"""
Entrypoint for trials.

Assuming execution engine is BaseExecutionEngine.
"""

from .execution.base import BaseExecutionEngine

if __name__ == '__main__':
    BaseExecutionEngine.trial_execute_graph()
