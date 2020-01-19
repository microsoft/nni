# Installation on Windows

## Installation

Anaconda or Miniconda is highly recommended.

### __Install NNI through pip__

  Prerequisite: `python(64-bit) >= 3.5`

  ```bash
  python -m pip install --upgrade nni
  ```

### __Install NNI through source code__

  Prerequisite: `python >=3.5`, `git`, `PowerShell`.

  ```bash
  git clone -b v0.8 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
  ```

## System requirements

Below are the minimum system requirements for NNI on Windows, Windows 10.1809 is well tested and recommend. Due to potential programming changes, the minimum system requirements for NNI may change over time.

||Minimum Requirements|Recommended Specifications|
|---|---|---|
|**Operating System**|Windows 10|Windows 10|
|**CPU**|Intel® Core™ i3 or AMD Phenom™ X3 8650|Intel® Core™ i5 or AMD Phenom™ II X3 or better|
|**GPU**|NVIDIA® GeForce® GTX 460|NVIDIA® GeForce® GTX 660 or better|
|**Memory**|4 GB RAM|6 GB RAM|
|**Storage**|30 GB available hare drive space|
|**Internet**|Boardband internet connection|
|**Resolution**|1024 x 768 minimum display resolution|


## Run NNI examples on Windows

When installation is done, use the **config_windows.yml** configuration to start an experiment for validation.

```bash
nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
```

For other examples you need to change trial command `python3` into `python` in each example YAML.

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