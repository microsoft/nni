from .base import BaseStrategy

try:
    has_tianshou = True
    import tianshou
except ImportError:
    has_tianshou = False


class PolicyBasedRL(BaseStrategy):

    def __init__(self):
        if not has_tianshou:
            raise ImportError('`tianshou` is required to run RL-based strategy. '
                              'Please use "pip install tianshou" to install it beforehand.')

    def run(self, base_model, applied_mutators):
        pass
