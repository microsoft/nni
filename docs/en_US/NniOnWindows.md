# NNI on Windows (experimental feature)

Currently we support local, remote and pai mode on Windows. Windows 10.1809 is well tested and recommended.

## **Installation on Windows**

  please refer to [Installation](Installation.md) for more details.

When these things are done, use the **config_windows.yml** configuration to start an experiment for validation.

```bash
nnictl create --config nni\examples\trials\mnist\config_windows.yml
```

For other examples you need to change trial command `python3` into `python` in each example YAML.

## **FAQ**

### simplejson failed when installing NNI

Make sure C++ 14.0 compiler installed.
>building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### Fail to run PowerShell when install NNI from source

If you run PowerShell script for the first time and did not set the execution policies for executing the script, you will meet this error below. Try to run PowerShell as administrator with this command first:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

>...cannot be loaded because running scripts is disabled on this system.

### Trial failed with missing DLL in command line or PowerShell

This error caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and fail to install SciPy. Using Anaconda or Miniconda with Python(64-bit) can solve it.
>ImportError: DLL load failed

### Trial failed on webUI

Please check the trial log file stderr for more details. If there is no such file and NNI is installed through pip, then you need to run PowerShell as administrator with this command first:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

If there is a stderr file, please check out. Two possible cases are as follows:

* forget to change the trial command `python3` into `python` in each experiment YAML.
* forget to install experiment dependencies such as TensorFlow, Keras and so on.

### Fail to use BOHB on Windows
Make sure C++ 14.0 compiler installed then try to run `nnictl package install --name=BOHB` to install the dependencies.

### Not supported tuner on Windows
SMAC is not supported currently, the specific reason can be referred to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).

Note:

* If there is any error like `Segmentation fault`, please refer to [FAQ](FAQ.md)
