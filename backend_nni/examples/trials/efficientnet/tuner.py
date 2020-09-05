from nni.gridsearch_tuner.gridsearch_tuner import GridSearchTuner


class FixedProductTuner(GridSearchTuner):
    """
    This tuner is essentially grid search, but it guarantees all the parameters with alpha * beta^2 * gamma^2 is
    approximately `product`.
    """

    def __init__(self, product):
        """
        :param product: the constant provided, should be 2 in EfficientNet-B1
        """
        super().__init__()
        self.product = product

    def _expand_parameters(self, para):
        """
        Filter out all qualified parameters
        """
        para = super()._expand_parameters(para)
        if all([key in para[0] for key in ["alpha", "beta", "gamma"]]):  # if this is an interested set
            ret_para = []
            for p in para:
                prod = p["alpha"] * (p["beta"] ** 2) * (p["gamma"] ** 2)
                if abs(prod - self.product) < 0.1:
                    ret_para.append(p)
            return ret_para
        return para
