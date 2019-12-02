# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import random
from copy import deepcopy
from functools import total_ordering
from queue import PriorityQueue

import numpy as np
from scipy.linalg import LinAlgError, cho_solve, cholesky, solve_triangular
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import rbf_kernel

from nni.utils import OptimizeMode
from nni.networkmorphism_tuner.graph_transformer import transform
from nni.networkmorphism_tuner.utils import Constant
from nni.networkmorphism_tuner.layers import is_layer


def layer_distance(a, b):
    """The distance between two layers."""
    # pylint: disable=unidiomatic-typecheck
    if not isinstance(a, type(b)):
        return 1.0
    if is_layer(a, "Conv"):
        att_diff = [
            (a.filters, b.filters),
            (a.kernel_size, b.kernel_size),
            (a.stride, b.stride),
        ]
        return attribute_difference(att_diff)
    if is_layer(a, "Pooling"):
        att_diff = [
            (a.padding, b.padding),
            (a.kernel_size, b.kernel_size),
            (a.stride, b.stride),
        ]
        return attribute_difference(att_diff)
    return 0.0


def attribute_difference(att_diff):
    ''' The attribute distance.
    '''

    ret = 0
    for a_value, b_value in att_diff:
        if max(a_value, b_value) == 0:
            ret += 0
        else:
            ret += abs(a_value - b_value) * 1.0 / max(a_value, b_value)
    return ret * 1.0 / len(att_diff)


def layers_distance(list_a, list_b):
    """The distance between the layers of two neural networks."""
    len_a = len(list_a)
    len_b = len(list_b)
    f = np.zeros((len_a + 1, len_b + 1))
    f[-1][-1] = 0
    for i in range(-1, len_a):
        f[i][-1] = i + 1
    for j in range(-1, len_b):
        f[-1][j] = j + 1
    for i in range(len_a):
        for j in range(len_b):
            f[i][j] = min(
                f[i][j - 1] + 1,
                f[i - 1][j] + 1,
                f[i - 1][j - 1] + layer_distance(list_a[i], list_b[j]),
            )
    return f[len_a - 1][len_b - 1]


def skip_connection_distance(a, b):
    """The distance between two skip-connections."""
    if a[2] != b[2]:
        return 1.0
    len_a = abs(a[1] - a[0])
    len_b = abs(b[1] - b[0])
    return (abs(a[0] - b[0]) + abs(len_a - len_b)) / \
        (max(a[0], b[0]) + max(len_a, len_b))


def skip_connections_distance(list_a, list_b):
    """The distance between the skip-connections of two neural networks."""
    distance_matrix = np.zeros((len(list_a), len(list_b)))
    for i, a in enumerate(list_a):
        for j, b in enumerate(list_b):
            distance_matrix[i][j] = skip_connection_distance(a, b)
    return distance_matrix[linear_sum_assignment(distance_matrix)].sum() + abs(
        len(list_a) - len(list_b)
    )


def edit_distance(x, y):
    """The distance between two neural networks.
    Args:
        x: An instance of NetworkDescriptor.
        y: An instance of NetworkDescriptor
    Returns:
        The edit-distance between x and y.
    """

    ret = layers_distance(x.layers, y.layers)
    ret += Constant.KERNEL_LAMBDA * skip_connections_distance(
        x.skip_connections, y.skip_connections
    )
    return ret


class IncrementalGaussianProcess:
    """Gaussian process regressor.
    Attributes:
        alpha: A hyperparameter.
    """

    def __init__(self):
        self.alpha = 1e-10
        self._distance_matrix = None
        self._x = None
        self._y = None
        self._first_fitted = False
        self._l_matrix = None
        self._alpha_vector = None

    @property
    def kernel_matrix(self):
        ''' Kernel matric.
        '''
        return self._distance_matrix

    def fit(self, train_x, train_y):
        """ Fit the regressor with more data.
        Args:
            train_x: A list of NetworkDescriptor.
            train_y: A list of metric values.
        """
        if self.first_fitted:
            self.incremental_fit(train_x, train_y)
        else:
            self.first_fit(train_x, train_y)

    def incremental_fit(self, train_x, train_y):
        """ Incrementally fit the regressor. """
        if not self._first_fitted:
            raise ValueError(
                "The first_fit function needs to be called first.")

        train_x, train_y = np.array(train_x), np.array(train_y)

        # Incrementally compute K
        up_right_k = edit_distance_matrix(self._x, train_x)
        down_left_k = np.transpose(up_right_k)
        down_right_k = edit_distance_matrix(train_x)
        up_k = np.concatenate((self._distance_matrix, up_right_k), axis=1)
        down_k = np.concatenate((down_left_k, down_right_k), axis=1)
        temp_distance_matrix = np.concatenate((up_k, down_k), axis=0)
        k_matrix = bourgain_embedding_matrix(temp_distance_matrix)
        diagonal = np.diag_indices_from(k_matrix)
        diagonal = (diagonal[0][-len(train_x):], diagonal[1][-len(train_x):])
        k_matrix[diagonal] += self.alpha

        try:
            self._l_matrix = cholesky(k_matrix, lower=True)  # Line 2
        except LinAlgError:
            return self

        self._x = np.concatenate((self._x, train_x), axis=0)
        self._y = np.concatenate((self._y, train_y), axis=0)
        self._distance_matrix = temp_distance_matrix

        self._alpha_vector = cho_solve(
            (self._l_matrix, True), self._y)  # Line 3

        return self

    @property
    def first_fitted(self):
        ''' if it is firsr fitted
        '''
        return self._first_fitted

    def first_fit(self, train_x, train_y):
        """ Fit the regressor for the first time. """
        train_x, train_y = np.array(train_x), np.array(train_y)

        self._x = np.copy(train_x)
        self._y = np.copy(train_y)

        self._distance_matrix = edit_distance_matrix(self._x)
        k_matrix = bourgain_embedding_matrix(self._distance_matrix)
        k_matrix[np.diag_indices_from(k_matrix)] += self.alpha

        self._l_matrix = cholesky(k_matrix, lower=True)  # Line 2

        self._alpha_vector = cho_solve(
            (self._l_matrix, True), self._y)  # Line 3

        self._first_fitted = True
        return self

    def predict(self, train_x):
        """Predict the result.
        Args:
            train_x: A list of NetworkDescriptor.
        Returns:
            y_mean: The predicted mean.
            y_std: The predicted standard deviation.
        """
        k_trans = np.exp(-np.power(edit_distance_matrix(train_x, self._x), 2))
        y_mean = k_trans.dot(self._alpha_vector)  # Line 4 (y_mean = f_star)

        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        l_inv = solve_triangular(
            self._l_matrix.T, np.eye(
                self._l_matrix.shape[0]))
        k_inv = l_inv.dot(l_inv.T)
        # Compute variance of predictive distribution
        y_var = np.ones(len(train_x), dtype=np.float)
        y_var -= np.einsum("ij,ij->i", np.dot(k_trans, k_inv), k_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            y_var[y_var_negative] = 0.0
        return y_mean, np.sqrt(y_var)


def edit_distance_matrix(train_x, train_y=None):
    """Calculate the edit distance.
    Args:
        train_x: A list of neural architectures.
        train_y: A list of neural architectures.
    Returns:
        An edit-distance matrix.
    """
    if train_y is None:
        ret = np.zeros((train_x.shape[0], train_x.shape[0]))
        for x_index, x in enumerate(train_x):
            for y_index, y in enumerate(train_x):
                if x_index == y_index:
                    ret[x_index][y_index] = 0
                elif x_index < y_index:
                    ret[x_index][y_index] = edit_distance(x, y)
                else:
                    ret[x_index][y_index] = ret[y_index][x_index]
        return ret
    ret = np.zeros((train_x.shape[0], train_y.shape[0]))
    for x_index, x in enumerate(train_x):
        for y_index, y in enumerate(train_y):
            ret[x_index][y_index] = edit_distance(x, y)
    return ret


def vector_distance(a, b):
    """The Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def bourgain_embedding_matrix(distance_matrix):
    """Use Bourgain algorithm to embed the neural architectures based on their edit-distance.
    Args:
        distance_matrix: A matrix of edit-distances.
    Returns:
        A matrix of distances after embedding.
    """
    distance_matrix = np.array(distance_matrix)
    n = len(distance_matrix)
    if n == 1:
        return distance_matrix
    np.random.seed(123)
    distort_elements = []
    r = range(n)
    k = int(math.ceil(math.log(n) / math.log(2) - 1))
    t = int(math.ceil(math.log(n)))
    counter = 0
    for i in range(0, k + 1):
        for t in range(t):
            s = np.random.choice(r, 2 ** i)
            for j in r:
                d = min([distance_matrix[j][s] for s in s])
                counter += len(s)
                if i == 0 and t == 0:
                    distort_elements.append([d])
                else:
                    distort_elements[j].append(d)
    return rbf_kernel(distort_elements, distort_elements)


class BayesianOptimizer:
    """ A Bayesian optimizer for neural architectures.
    Attributes:
        searcher: The Searcher which is calling the Bayesian optimizer.
        t_min: The minimum temperature for simulated annealing.
        metric: An instance of the Metric subclasses.
        gpr: A GaussianProcessRegressor for bayesian optimization.
        beta: The beta in acquisition function. (refer to our paper)
        search_tree: The network morphism search tree.
    """

    def __init__(self, searcher, t_min, optimizemode, beta=None):
        self.searcher = searcher
        self.t_min = t_min
        self.optimizemode = optimizemode
        self.gpr = IncrementalGaussianProcess()
        self.beta = beta if beta is not None else Constant.BETA
        self.search_tree = SearchTree()

    def fit(self, x_queue, y_queue):
        """ Fit the optimizer with new architectures and performances.
        Args:
            x_queue: A list of NetworkDescriptor.
            y_queue: A list of metric values.
        """
        self.gpr.fit(x_queue, y_queue)

    def generate(self, descriptors):
        """Generate new architecture.
        Args:
            descriptors: All the searched neural architectures.
        Returns:
            graph: An instance of Graph. A morphed neural network with weights.
            father_id: The father node ID in the search tree.
        """
        model_ids = self.search_tree.adj_list.keys()

        target_graph = None
        father_id = None
        descriptors = deepcopy(descriptors)
        elem_class = Elem
        if self.optimizemode is OptimizeMode.Maximize:
            elem_class = ReverseElem

        # Initialize the priority queue.
        pq = PriorityQueue()
        temp_list = []
        for model_id in model_ids:
            metric_value = self.searcher.get_metric_value_by_id(model_id)
            temp_list.append((metric_value, model_id))
        temp_list = sorted(temp_list)
        for metric_value, model_id in temp_list:
            graph = self.searcher.load_model_by_id(model_id)
            graph.clear_operation_history()
            graph.clear_weights()
            pq.put(elem_class(metric_value, model_id, graph))

        t = 1.0
        t_min = self.t_min
        alpha = 0.9
        opt_acq = self._get_init_opt_acq_value()
        while not pq.empty() and t > t_min:
            elem = pq.get()
            if self.optimizemode is OptimizeMode.Maximize:
                temp_exp = min((elem.metric_value - opt_acq) / t, 1.0)
            else:
                temp_exp = min((opt_acq - elem.metric_value) / t, 1.0)
            ap = math.exp(temp_exp)
            if ap >= random.uniform(0, 1):
                for temp_graph in transform(elem.graph):
                    if contain(descriptors, temp_graph.extract_descriptor()):
                        continue

                    temp_acq_value = self.acq(temp_graph)
                    pq.put(
                        elem_class(
                            temp_acq_value,
                            elem.father_id,
                            temp_graph))
                    descriptors.append(temp_graph.extract_descriptor())
                    if self._accept_new_acq_value(opt_acq, temp_acq_value):
                        opt_acq = temp_acq_value
                        father_id = elem.father_id
                        target_graph = deepcopy(temp_graph)
            t *= alpha

        # Did not found a not duplicated architecture
        if father_id is None:
            return None, None
        nm_graph = self.searcher.load_model_by_id(father_id)
        for args in target_graph.operation_history:
            getattr(nm_graph, args[0])(*list(args[1:]))
        return nm_graph, father_id

    def acq(self, graph):
        ''' estimate the value of generated graph
        '''
        mean, std = self.gpr.predict(np.array([graph.extract_descriptor()]))
        if self.optimizemode is OptimizeMode.Maximize:
            return mean + self.beta * std
        return mean - self.beta * std

    def _get_init_opt_acq_value(self):
        if self.optimizemode is OptimizeMode.Maximize:
            return -np.inf
        return np.inf

    def _accept_new_acq_value(self, opt_acq, temp_acq_value):
        if temp_acq_value > opt_acq and self.optimizemode is OptimizeMode.Maximize:
            return True
        if temp_acq_value < opt_acq and not self.optimizemode is OptimizeMode.Maximize:
            return True
        return False

    def add_child(self, father_id, model_id):
        ''' add child to the search tree
        Arguments:
            father_id {int} -- father id
            model_id {int} -- model id
        '''

        self.search_tree.add_child(father_id, model_id)


@total_ordering
class Elem:
    """Elements to be sorted according to metric value."""

    def __init__(self, metric_value, father_id, graph):
        self.father_id = father_id
        self.graph = graph
        self.metric_value = metric_value

    def __eq__(self, other):
        return self.metric_value == other.metric_value

    def __lt__(self, other):
        return self.metric_value < other.metric_value


class ReverseElem(Elem):
    """Elements to be reversely sorted according to metric value."""

    def __lt__(self, other):
        return self.metric_value > other.metric_value


def contain(descriptors, target_descriptor):
    """Check if the target descriptor is in the descriptors."""
    for descriptor in descriptors:
        if edit_distance(descriptor, target_descriptor) < 1e-5:
            return True
    return False


class SearchTree:
    """The network morphism search tree."""

    def __init__(self):
        self.root = None
        self.adj_list = {}

    def add_child(self, u, v):
        ''' add child to search tree itself.
        Arguments:
            u {int} -- father id
            v {int} --  child id
        '''

        if u == -1:
            self.root = v
            self.adj_list[v] = []
            return
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if v not in self.adj_list:
            self.adj_list[v] = []

    def get_dict(self, u=None):
        """ A recursive function to return the content of the tree in a dict."""
        if u is None:
            return self.get_dict(self.root)
        children = []
        for v in self.adj_list[u]:
            children.append(self.get_dict(v))
        ret = {"name": u, "children": children}
        return ret
