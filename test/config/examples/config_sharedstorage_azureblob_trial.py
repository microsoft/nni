"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
import os
import logging
import nni
logger = logging.getLogger('mnist_AutoML')
if __name__ == '__main__':
    try:
        logger.debug(os.environ.get('NNI_OUTPUT_DIR'))
        filename = os.path.join(os.environ.get('NNI_OUTPUT_DIR'), 'checkingfile.txt')
        f = open(filename, "a")
        f.write("this is the checking file.")
        f.close()
    except Exception as exception:
        logger.exception(exception)
        raise
