# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
utility functions and classes for GPTuner
"""

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def _match_val_type(vals, bounds):
    """
    Update values in the array, to match their corresponding type, make sure the value is legal.

    Parameters
    ----------
    vals : numpy array
        values of parameters
    bounds : numpy array
        list of dictionary which stores parameters names and legal values.

    Returns
    -------
    vals_new : list
        The closest legal value to the original value
    """
    vals_new = []

    for i, bound in enumerate(bounds):
        _type = bound['_type']
        if _type == "choice":
            # Find the closest integer in the array, vals_bounds
            # pylint: disable=cell-var-from-loop
            vals_new.append(min(bound['_value'], key=lambda x: abs(x - vals[i])))
        elif _type in ['quniform', 'randint']:
            vals_new.append(np.around(vals[i]))
        else:
            vals_new.append(vals[i])

    return vals_new


def acq_max(f_acq, gp, y_max, bounds, space, num_warmup, num_starting_points):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling ``num_warmup`` points at random,
    and then running L-BFGS-B from ``num_starting_points`` random starting points.

    Parameters
    ----------
    f_acq : UtilityFunction.utility
        The acquisition function object that return its point-wise value.

    gp : GaussianProcessRegressor
        A gaussian process fitted to the relevant data.

    y_max : float
        The current maximum known value of the target function.

    bounds : numpy array
        The variables bounds to limit the search of the acq max.

    num_warmup : int
        number of times to randomly sample the aquisition function

    num_starting_points : int
        number of times to run scipy.minimize

    Returns
    -------
    numpy array
        The parameter which achieves max of the acquisition function.
    """

    # Warm up with random points
    x_tries = [space.random_sample()
               for _ in range(int(num_warmup))]
    ys = f_acq(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()


    # Explore the parameter space more throughly
    x_seeds = [space.random_sample() for _ in range(int(num_starting_points))]

    bounds_minmax = np.array(
        [[bound['_value'][0], bound['_value'][-1]] for bound in bounds])

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -f_acq(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds_minmax,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = _match_val_type(res.x, bounds)
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds_minmax[:, 0], bounds_minmax[:, 1])


class UtilityFunction():
    """
    A class to compute different acquisition function values.

    Parameters
    ----------
    kind : string
        specification of utility function to use
    kappa : float
        parameter usedd for 'ucb' acquisition function
    xi : float
        parameter usedd for 'ei' and 'poi' acquisition function
    """

    def __init__(self, kind, kappa, xi):
        self._kappa = kappa
        self._xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                "{} has not been implemented, " \
                "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        self._kind = kind

    def utility(self, x, gp, y_max):
        """
        return utility function

        Parameters
        ----------
        x : numpy array
            parameters
        gp : GaussianProcessRegressor
        y_max : float
            maximum target value observed so far

        Returns
        -------
        function
            return corresponding function, return None if parameter is illegal
        """
        if self._kind == 'ucb':
            return self._ucb(x, gp, self._kappa)
        if self._kind == 'ei':
            return self._ei(x, gp, y_max, self._xi)
        if self._kind == 'poi':
            return self._poi(x, gp, y_max, self._xi)
        return None

    @staticmethod
    def _ucb(x, gp, kappa):
        """
        Upper Confidence Bound (UCB) utility function

        Parameters
        ----------
        x : numpy array
            parameters
        gp : GaussianProcessRegressor
        kappa : float

        Returns
        -------
        float
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        """
        Expected Improvement (EI) utility function

        Parameters
        ----------
        x : numpy array
            parameters
        gp : GaussianProcessRegressor
        y_max : float
            maximum target value observed so far
        xi : float

        Returns
        -------
        float
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        """
        Possibility Of Improvement (POI) utility function

        Parameters
        ----------
        x : numpy array
            parameters
        gp : GaussianProcessRegressor
        y_max : float
            maximum target value observed so far
        xi : float

        Returns
        -------
        float
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)
