import hashlib

import numpy as np

LABEL2ID = {
    'input': -1,
    'output': -2,
    'conv3x3-bn-relu': 0,
    'conv1x1-bn-relu': 1,
    'maxpool3x3': 2
}


def labeling_from_architecture(architecture, vertices):
    return ['input'] + [architecture['op{}'.format(i)] for i in range(1, vertices - 1)] + ['output']


def adjancency_matrix_from_architecture(architecture, vertices):
    matrix = np.zeros((vertices, vertices), dtype=np.bool)
    for i in range(1, vertices):
        for k in architecture['input{}'.format(i)]:
            matrix[k, i] = 1
    return matrix


def nasbench_format_to_architecture_repr(adjacency_matrix, labeling):
    num_vertices = adjacency_matrix.shape[0]
    assert len(labeling) == num_vertices
    architecture = {}
    for i in range(1, num_vertices - 1):
        architecture['op{}'.format(i)] = labeling[i]
        assert labeling[i] not in ['input', 'output']
    for i in range(1, num_vertices):
        architecture['input{}'.format(i)] = [k for k in range(i) if adjacency_matrix[k, i]]
    return num_vertices, architecture


def hash_module(architecture, vertices):
    """
    Computes a graph-invariance MD5 hash of the matrix and label pair.
    Imported from NAS-Bench-101 repo.

    Parameters
    ----------
    matrix : np.ndarray
        Square upper-triangular adjacency matrix.
    labeling : list of int
        Labels of length equal to both dimensions of matrix.

    Returns
    -------
        MD5 hash of the matrix and labeling.
    """
    labeling = labeling_from_architecture(architecture, vertices)
    labeling = [LABEL2ID[t] for t in labeling]
    matrix = adjancency_matrix_from_architecture(architecture, vertices)
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint
