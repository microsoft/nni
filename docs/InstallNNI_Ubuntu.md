**Install NNI on Ubuntu**
===

## **Installation**
* __Dependencies__

      python >= 3.5
      git
      wget

    python pip should also be correctly installed. You could use "which pip" or "pip -V" to check in Linux.
    
    * Note: we don't support virtual environment in current releases.

* __Install NNI through pip__

      pip3 install -v --user git+https://github.com/Microsoft/nni.git@v0.1
      source ~/.bashrc

* __Install NNI through source code__
   
      git clone -b v0.1 https://github.com/Microsoft/nni.git
      cd nni
      chmod +x install.sh
      source install.sh


## Learn More
* [Get started](GetStarted.md)
### **How to**
* [Use command line tool nnictl](InstallNNI_Ubuntu.md)
* [Use NNIBoard](InstallNNI_Ubuntu.md)
* [Define search space](InstallNNI_Ubuntu.md)
* [Use NNI sdk](InstallNNI_Ubuntu.md)
* [Config an experiment](InstallNNI_Ubuntu.md)
* [Use annotation](InstallNNI_Ubuntu.md)
* [Debug](InstallNNI_Ubuntu.md)
### **Tutorials**
* [Try different tuners and assessors]()
* [How to run an experiment on local (with multiple GPUs)?]()
* [How to run an experiment on multiple machines?]()
* [How to run an experiment on OpenPAI?]()
* [How to run an experiment on K8S services?]()
* [Implement a customized tuner]()
* [Implement a customized assessor]()
* [Implement a custmoized weight sharing algorithm]()
* [How to integrate NNI with your own custmoized training service]()
### **Best practice**
1. [Create and Run your first experiement at local with NNI API](tutorial_1_CR_exp_local_api.md)
2. [Run an experiment on multiple machines](tutorial_2_RemoteMachineMode.md)
3. [Compare different AutoML algorithms] - *coming soon*
4. [Serve NNI as a capability of a ML Platform] - *coming soon*
