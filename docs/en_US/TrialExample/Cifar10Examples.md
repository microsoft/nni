# CIFAR-10 examples

## Overview

[CIFAR-10][3] classification is a common benchmark problem in machine learning. The CIFAR-10 dataset is the collection of images. It is one of the most widely used datasets for machine learning research which contains 60,000 32x32 color images in 10 different classes. Thus, we use CIFAR-10 classification as an example to introduce NNI usage.

### **Goals**

As we all know, the choice of model optimizer is directly affects the performance of the final metrics. The goal of this tutorial is to **tune a better performace optimizer** to train a relatively small convolutional neural network (CNN) for recognizing images.

In this example, we have selected the following common deep learning optimizer:

> "SGD", "Adadelta", "Adagrad", "Adam", "Adamax"

### **Experimental**

#### Preparations

This example requires PyTorch. PyTorch install package should be chosen based on python version and cuda version.

Here is an example of the environment python==3.5 and cuda == 8.0, then using the following commands to install [PyTorch][2]:

```bash
python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
python3 -m pip install torchvision
```

#### CIFAR-10 with NNI

**Search Space**

As we stated in the target, we target to find out the best `optimizer` for training CIFAR-10 classification. When using different optimizers, we also need to adjust `learning rates` and `network structure` accordingly. so we chose these three parameters as hyperparameters and write the following search space.

```json
{
    "lr":{"_type":"choice", "_value":[0.1, 0.01, 0.001, 0.0001]},
    "optimizer":{"_type":"choice", "_value":["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"]},
    "model":{"_type":"choice", "_value":["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "senet18"]}
}
```

*Implemented code directory: [search_space.json][8]*

**Trial**

The code for CNN training of each hyperparameters set, paying particular attention to the following points are specific for NNI:

* Use `nni.get_next_parameter()` to get next training hyperparameter set.
* Use `nni.report_intermediate_result(acc)` to report the intermedian result after finish each epoch.
* Use `nni.report_final_result(acc)` to report the final result before the trial end.

*Implemented code directory: [main.py][9]*

You can also use your previous code directly, refer to [How to define a trial][5] for modify.

**Config**

Here is the example of running this experiment on local(with multiple GPUs):

code directory: [examples/trials/cifar10_pytorch/config.yml][6]

Here is the example of running this experiment on OpenPAI:

code directory: [examples/trials/cifar10_pytorch/config_pai.yml][7]

*The complete examples we have implemented: [examples/trials/cifar10_pytorch/][1]*

#### Launch the experiment

We are ready for the experiment, let's now **run the config.yml file from your command line to start the experiment**.

```bash
nnictl create --config nni/examples/trials/cifar10_pytorch/config.yml
```

[1]: https://github.com/Microsoft/nni/tree/v1.9/examples/trials/cifar10_pytorch
[2]: https://pytorch.org/
[3]: https://www.cs.toronto.edu/~kriz/cifar.html
[4]: https://github.com/Microsoft/nni/tree/v1.9/examples/trials/cifar10_pytorch
[5]: Trials.md
[6]: https://github.com/Microsoft/nni/blob/v1.9/examples/trials/cifar10_pytorch/config.yml
[7]: https://github.com/Microsoft/nni/blob/v1.9/examples/trials/cifar10_pytorch/config_pai.yml
[8]: https://github.com/Microsoft/nni/blob/v1.9/examples/trials/cifar10_pytorch/search_space.json
[9]: https://github.com/Microsoft/nni/blob/v1.9/examples/trials/cifar10_pytorch/main.py
