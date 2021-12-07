import nni.retiarii.nn.pytorch as nn
from nni.retiarii import basic_unit


@basic_unit
class ImportTest(nn.Module):
    def __init__(self, foo, bar):
        super().__init__()
        self.foo = nn.Linear(foo, 3)
        self.bar = nn.Dropout(bar)

    def __eq__(self, other):
        return self.foo.in_features == other.foo.in_features and self.bar.p == other.bar.p
