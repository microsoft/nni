# Install on Windows

## Installation

Anaconda or Miniconda is highly recommended to manage multiple Python environments.

### Install NNI through pip

  Prerequisites: `python 64-bit >= 3.5`

  ```bash
  python -m pip install --upgrade nni
  ```

### Install NNI through source code

  If you are interested on special or latest code version, you can install NNI through source code.

  Prerequisites: `python 64-bit >=3.5`, `git`, `PowerShell`.

  ```bash
  git clone -b v1.3 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
  ```

## Verify installation

The following example is built on TensorFlow 1.x. Make sure **TensorFlow 1.x is used** when running it.

* Download the examples via clone the source code.

  ```bash
  git clone -b v1.3 https://github.com/Microsoft/nni.git
  ```

* Run the MNIST example.

  ```bash
  nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
  ```

  Note: for other examples you need to change trial command `python3` to `python` in each example YAML, if python3 is called through `python` on your machine.

* Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. You can explore the experiment using the `Web UI url`.

```text
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
-----------------------------------------------------------------------

You can use these commands to get more information about the experiment
-----------------------------------------------------------------------
         commands                       description
1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
-----------------------------------------------------------------------
```

* Open the `Web UI url` in your browser, you can view detail information of the experiment and all the submitted trial jobs as shown below. [Here](../Tutorial/WebUI.md) are more Web UI pages.

![overview](../../img/webui_overview_page.png)

![detail](../../img/webui_trialdetail_page.png)

## System requirements

Below are the minimum system requirements for NNI on Windows, Windows 10.1809 is well tested and recommend. Due to potential programming changes, the minimum system requirements for NNI may change over time.

|                      | Recommended                                    | Minimum                                |
| -------------------- | ---------------------------------------------- | -------------------------------------- |
| **Operating System** | Windows 10 1809 or above                       |
| **CPU**              | Intel® Core™ i5 or AMD Phenom™ II X3 or better | Intel® Core™ i3 or AMD Phenom™ X3 8650 |
| **GPU**              | NVIDIA® GeForce® GTX 660 or better             | NVIDIA® GeForce® GTX 460               |
| **Memory**           | 6 GB RAM                                       | 4 GB RAM                               |
| **Storage**          | 30 GB available hare drive space               |
| **Internet**         | Boardband internet connection                  |
| **Resolution**       | 1024 x 768 minimum display resolution          |

## FAQ

### simplejson failed when installing NNI

Make sure C++ 14.0 compiler installed.
>building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### Trial failed with missing DLL in command line or PowerShell

This error caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and fail to install SciPy. Using Anaconda or Miniconda with Python(64-bit) can solve it.
>ImportError: DLL load failed

### Trial failed on webUI

Please check the trial log file stderr for more details.

If there is a stderr file, please check out. Two possible cases are as follows:

* forget to change the trial command `python3` into `python` in each experiment YAML.
* forget to install experiment dependencies such as TensorFlow, Keras and so on.

### Fail to use BOHB on Windows
Make sure C++ 14.0 compiler installed then try to run `nnictl package install --name=BOHB` to install the dependencies.

### Not supported tuner on Windows
SMAC is not supported currently, the specific reason can be referred to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).

### Use a Windows server as a remote worker
Currently you can't.

Note:

* If there is any error like `Segmentation fault`, please refer to [FAQ](FAQ.md)


## Further reading

* [Overview](../Overview.md)
* [Use command line tool nnictl](Nnictl.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](../TrainingService/LocalMode.md)
* [How to run an experiment on multiple machines?](../TrainingService/RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](../TrainingService/PaiMode.md)
* [How to run an experiment on Kubernetes through Kubeflow?](../TrainingService/KubeflowMode.md)
* [How to run an experiment on Kubernetes through FrameworkController?](../TrainingService/FrameworkControllerMode.md)