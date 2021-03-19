import os
import torchvision.datasets.mnist as mnist

file_name = mnist.__file__
dummy_file_name = os.path.join(os.path.dirname(file_name), 'mnist_dummy.py')
with open(file_name, 'r') as fr, open(dummy_file_name, 'w') as fw:
    origin_text = fr.read()
    mnist_head = origin_text.find('class MNIST(')
    reasource_head = origin_text.find('resources = [', mnist_head)
    reasource_tail = origin_text.find(']\n', reasource_head)
    top = origin_text[:reasource_head]
    reasource = "resources = [('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')]\n"
    bottom = origin_text[reasource_tail + 2:]
    fw.write(top)
    fw.write(reasource)
    fw.write(bottom)

if os.path.exists(dummy_file_name):
    os.remove(file_name)
    os.rename(dummy_file_name, file_name)
