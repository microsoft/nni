from nni.gridsearch_tuner.gridsearch_tuner import GridSearchTuner


class FixedProductTuner(GridSearchTuner):

    def __init__(self, product):
        super().__init__()
        self.product = product

    def expand_parameters(self, para):
        para = super().expand_parameters(para)
        if all([key in para[0] for key in ["alpha", "beta", "gamma"]]):
            ret_para = []
            for p in para:
                prod = p["alpha"] * (p["beta"] ** 2) * (p["gamma"] ** 2)
                if abs(prod - self.product) < 0.1:
                    ret_para.append(p)
            return ret_para
        return para
