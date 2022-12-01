import pytest
import nni

from nni.trial import overwrite_intermediate_seq
from nni.runtime.trial_command_channel import (
    TrialCommandChannel, set_default_trial_command_channel,
    get_default_trial_command_channel
)


def test_standalone(caplog):
    overwrite_intermediate_seq(0)

    with pytest.warns(RuntimeWarning) as record:
        nni.get_next_parameter()

    assert len(record) == 1
    assert 'Running trial code without runtime.' in record[0].message.args[0]
    nni.report_intermediate_result(123)
    assert 'Intermediate result: 123  (Index 0)' in caplog.text
    nni.report_intermediate_result(456)
    assert 'Intermediate result: 456  (Index 1)' in caplog.text
    nni.report_final_result(123)
    assert 'Final result: 123' in caplog.text


def test_customized_channel():
    try:
        _default_channel = get_default_trial_command_channel()

        class TestChannel(TrialCommandChannel):
            def receive_parameter(self):
                return {'parameter_id': 123, 'parameters': {'x': 0}}

            def send_metric(self, *args, **kwargs):
                assert kwargs.get('type') == 'FINAL'
                assert kwargs.get('parameter_id') == 123
                assert kwargs.get('value') == 456
                self._received = True

        set_default_trial_command_channel(TestChannel())
        assert nni.get_next_parameter()['x'] == 0
        nni.report_final_result(456)
        assert get_default_trial_command_channel()._received

    finally:
        set_default_trial_command_channel(_default_channel)
