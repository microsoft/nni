# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tool class to hold the param-space coordinates (X) and target values (Y).
"""

import numpy as np
import nni.parameter_expressions as parameter_expressions


def _hashable(params):
    """
    Transform list params to tuple format. Ensure that an point is hashable by a python dict.

    Parameters
    ----------
    params : numpy array
        array format of parameters

    Returns
    -------
    tuple
        tuple format of parameters
    """
    return tuple(map(float, params))


class TargetSpace():
    """
    Holds the param-space coordinates (X) and target values (Y)

    Parameters
    ----------
    pbounds : dict
        Dictionary with parameters names and legal values.

    random_state : int, RandomState, or None
        optionally specify a seed for a random number generator, by default None.
    """

    def __init__(self, pbounds, random_state=None):
        self._random_state = random_state

        # Get the name of the parameters
        self._keys = sorted(pbounds)

        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])]
        )

        # check values type
        for _bound in self._bounds:
            if _bound['_type'] == 'choice':
                try:
                    [float(val) for val in _bound['_value']]
                except ValueError:
                    raise ValueError("GP Tuner supports only numerical values")

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, params):
        """
        check if a parameter is already registered

        Parameters
        ----------
        params : numpy array

        Returns
        -------
        bool
            True if the parameter is already registered, else false
        """
        return _hashable(params) in self._cache

    def len(self):
        """
        length of registered params and targets

        Returns
        -------
        int
        """
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def params(self):
        """
        registered parameters

        Returns
        -------
        numpy array
        """
        return self._params

    @property
    def target(self):
        """
        registered target values

        Returns
        -------
        numpy array
        """
        return self._target

    @property
    def dim(self):
        """
        dimension of parameters

        Returns
        -------
        int
        """
        return len(self._keys)

    @property
    def keys(self):
        """
        keys of parameters

        Returns
        -------
        numpy array
        """
        return self._keys

    @property
    def bounds(self):
        """
        bounds of parameters

        Returns
        -------
        numpy array
        """
        return self._bounds

    def params_to_array(self, params):
        """
        dict to array

        Parameters
        ----------
        params : dict
            dict format of parameters

        Returns
        -------
        numpy array
            array format of parameters
        """
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        """
        array to dict

        maintain int type if the paramters is defined as int in search_space.json
        Parameters
        ----------
        x : numpy array
            array format of parameters

        Returns
        -------
        dict
            dict format of parameters
        """
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(self.dim)
            )

        params = {}
        for i, _bound in enumerate(self._bounds):
            if _bound['_type'] == 'choice' and all(isinstance(val, int) for val in _bound['_value']):
                params.update({self.keys[i]: int(x[i])})
            elif _bound['_type'] in ['randint']:
                params.update({self.keys[i]: int(x[i])})
            else:
                params.update({self.keys[i]:  x[i]})

        return params

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        params : dict
            parameters

        target : float
            target function value
        """

        x = self.params_to_array(params)
        if x in self:
            print('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def random_sample(self):
        """
        Creates a random point within the bounds of the space.

        Returns
        -------
        numpy array
            one groupe of parameter
        """
        params = np.empty(self.dim)
        for col, _bound in enumerate(self._bounds):
            if _bound['_type'] == 'choice':
                params[col] = parameter_expressions.choice(
                    _bound['_value'], self._random_state)
            elif _bound['_type'] == 'randint':
                params[col] = self._random_state.randint(
                    _bound['_value'][0], _bound['_value'][1], size=1)
            elif _bound['_type'] == 'uniform':
                params[col] = parameter_expressions.uniform(
                    _bound['_value'][0], _bound['_value'][1], self._random_state)
            elif _bound['_type'] == 'quniform':
                params[col] = parameter_expressions.quniform(
                    _bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self._random_state)
            elif _bound['_type'] == 'loguniform':
                params[col] = parameter_expressions.loguniform(
                    _bound['_value'][0], _bound['_value'][1], self._random_state)
            elif _bound['_type'] == 'qloguniform':
                params[col] = parameter_expressions.qloguniform(
                    _bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self._random_state)

        return params

    def max(self):
        """
        Get maximum target value found and its corresponding parameters.

        Returns
        -------
        dict
            target value and parameters, empty dict if nothing registered
        """
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """
        Get all target values found and corresponding parameters.

        Returns
        -------
        list
            a list of target values and their corresponding parameters
        """
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]
