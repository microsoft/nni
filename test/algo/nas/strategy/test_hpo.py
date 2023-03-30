import pytest
from nni.nas.strategy import TPE

from ut.nas.strategy.test_sanity import named_model_space, engine


def test_tpe_sanity(named_model_space, engine):
    # put here because it takes long for some reason.
    name, model_space = named_model_space
    if name != 'base':
        pytest.skip('TPE strategy only supports basic test case.')
    strategy = TPE()
    assert repr(strategy) == f'TPE(tuner={strategy.tuner!r})'
    strategy(model_space, engine)
    assert next(strategy.list_models()).metric == 1.0

    state_dict = strategy.state_dict()
    strategy2 = TPE()
    strategy2.load_state_dict(state_dict)
    strategy2(model_space, engine)
