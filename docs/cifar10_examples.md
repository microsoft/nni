# CIFAR10 examples

The CIFAR-10 dataset is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. There are 6,000 images of each class. Thus, we use CIFAR10 as example to introduce using NNI in the field of computer vision.

**cifar10 with NNI**

This example requires pytorch. pytorch install package should be chosen based on python version and cuda version.

Here is an example of the environment python==3.5 and cuda == 8.0, then using the following commands to install pytorch:

```bash
python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
python3 -m pip install torchvision
```

`code directory`: [examples/trials/cifar10_pytorch/][1]

[1]: https://github.com/Microsoft/nni/tree/master/examples/trials/cifar10_pytorch