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


## Further reading
* [NNI Overview](docs/Overview.md)
* [Tutorial: Create and Run your first experiement at local with NNI API](docs/tutorial_1_CR_exp_local_api.md)
* [Tutorial: Run an experiment on multiple machines](docs/tutorial_2_RemoteMachineMode.md)
* [How to write a Trial?](howto_1_WriteTrial.md)
* [How to write a customized Tuner?](howto_2_CustomizedTuner.md)
* [How to write a customized Assessor?](../examples/assessors/README.md)
* [How to enable Assessor for early stop in an experiment?](EnableAssessor.md)

* [Tutorial: Compare different AutoML algorithms] - *coming soon*
* [Tutorial: Serve NNI as a capability of a ML Platform] - *coming soon*
* [How to write an experiment?] - *coming soon*
* [How to resume an experiment?] - *coming soon*