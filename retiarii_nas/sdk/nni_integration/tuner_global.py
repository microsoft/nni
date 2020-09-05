from threading import Thread

from .. import utils


_tuner_instance = None

def init(tuner):
    global _tuner_instance
    _tuner_instance = tuner
    strategy = utils.import_(utils.experiment_config()['strategy.scheduler'])
    Thread(target=strategy).start()

def get_tuner():
    return _tuner_instance
