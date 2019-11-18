from nni.nas.pytorch.mutables import MutableScope


class DartsNode(MutableScope):
    """
    At most `limitation` choice is activated in a `DartsNode` when exporting.
    """

    def __init__(self, key, limitation):
        super().__init__(key)
        self.limitation = limitation
