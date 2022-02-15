import nni
import functools

@nni.trace
class Foo:
    def __init__(self, *args, **kwargs):
        pass

import pickle

print(pickle.loads(pickle.dumps(Foo(1, 2, k=3))))

# from nni.retiarii.evaluator.pytorch.lightning import _ClassificationModule

# pickle.dumps(_ClassificationModule())

