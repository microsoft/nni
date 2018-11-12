**Install NNI on Ubuntu**
===

## **Installation**
* __Dependencies__

      python >= 3.5
      git
      wget

    python pip should also be correctly installed. You could use "python3 -m pip -v" to check in Linux.

* __Install NNI through pip__

      python3 -m pip install --user --upgrade nni

* __Install NNI through source code__
   
      git clone -b v0.3.4 https://github.com/Microsoft/nni.git
      cd nni
      source install.sh

* __Install NNI in docker image__

    You can also install NNI in a docker image. Please follow the instructions [here](../deployment/docker/README.md) to build NNI docker image. The NNI docker image can also be retrieved from Docker Hub through the command `docker pull msranni/nni:latest`.

## Further reading
* [Overview](Overview.md)
* [Use command line tool nnictl](NNICTLDOC.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](tutorial_2_RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)
