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
