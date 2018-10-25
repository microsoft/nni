**Install NNI on Ubuntu**
===

## **Installation**
* __Dependencies__

      python >= 3.5
      git
      wget

    python pip should also be correctly installed. You could use "which pip" or "pip -V" to check in Linux.

* __Install NNI through pip__

      pip3 install -v --user git+https://github.com/Microsoft/nni.git@v0.2
      source ~/.bashrc

* __Install NNI through source code__
   
      git clone -b v0.1 https://github.com/Microsoft/nni.git
      cd nni
      chmod +x install.sh
      source install.sh


## Further reading
* [Overview](Overview.md)
* [Use command line tool nnictl](NNICTLDOC.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](tutorial_2_RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)
