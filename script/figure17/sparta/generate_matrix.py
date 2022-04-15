import random

def generate_sparse_matrix(dim1, dim2, sparsity=0, file_name=None):
    random.seed(a=3)
    # row major
    size = dim1 * dim2
    matrix = []
    for i in range(size):
        if random.randint(0, 99) >= sparsity:
            matrix.append(random.randint(1, 20))
        else:
            matrix.append(0)
    if file_name is not None:
        with open(file_name, 'w') as fp:
            for item in matrix:
                fp.write("%s\n" % item)

    return matrix

def generate_sparse_matrix_float(dim1, dim2, sparsity=0, file_name=None):
    random.seed(a=3)
    # row major
    size = dim1 * dim2
    matrix = []
    for i in range(size):
        if random.randint(0, 99) >= sparsity:
            matrix.append(float(random.randint(1, 20)))
        else:
            matrix.append(0)
    if file_name is not None:
        with open(file_name, 'w') as fp:
            for item in matrix:
                fp.write("%.1s\n" % item)

    return matrix

def load_sparse_matrix_float(file_name):
    matrix = []
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            matrix.append(float(line.rstrip('\n')))
    return matrix

def load_tesa_matrix_float(file_name):
    matrix = []
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            items = line.rstrip('\n').split(',')
            for item in items:
                if item == "-0":
                    item = "0"
                matrix.append(float(item))
    return matrix