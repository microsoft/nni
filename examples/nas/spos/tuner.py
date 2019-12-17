from nni.nas.pytorch.spos import SPOSEvolution

from network import ShuffleNetV2OneShot


class EvolutionWithFlops(SPOSEvolution):
    def __init__(self, flops_limit=330E6, **kwargs):
        super().__init__(**kwargs)
        self.model = ShuffleNetV2OneShot()
        self.flops_limit = flops_limit

    def _is_legal(self, cand):
        if not super()._is_legal(cand):
            return False
        if self.model.get_candidate_flops(cand) > self.flops_limit:
            return False
        return True
