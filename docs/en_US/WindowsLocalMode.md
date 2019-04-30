# Windows Local Mode (experimental feature)

Currently we only support local mode on Windows. Windows 10.1809 is well tested and recommended.

## **Installation on Windows**

  **Anaconda python(64-bit) is highly recommended.**  

When you use PowerShell to run script for the first time, you need **run PowerShell as administrator** with this command:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

* __Install NNI through pip__

  Prerequisite: `python(64-bit) >= 3.5`

  ```bash
  python -m pip install --upgrade nni
  ```

* __Install NNI through source code__

  Prerequisite: `python >=3.5`, `git`, `powershell`

  ```bash
  git clone -b v0.7 https://github.com/Microsoft/nni.git
  cd nni
  powershell ./install.ps1
  ```

When these things are done, run the **config_windows.yml** file from your command line to start the experiment.

```bash
nnictl create --config nni/examples/trials/mnist/config_windows.yml
```

For other examples you need to change trial command `python3` into `python` in each example YAML.

## **Frequent met errors and answers**

### simplejson failed when installing NNI

Make sure C++ 14.0 compiler installed.
>building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### Fail to run PowerShell when install NNI from source
If you run PowerShell script for the first time and did not set the execution policies for executing the script, you will meet this error below. Try to run PowerShell as administrator with this command first:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

>...cannot be loaded because running scripts is disabled on this system.

### Trial failed with missing DLL in cmd or PowerShell

This error caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and fail to install SciPy. Anaconda python is highly recommended. If you use official python, make sure you have one of `Visual Studio`, `MATLAB`, `MKL` and `Intel Distribution for Python` installed on Windows before running NNI. If not, try to install one of products above or Anaconda python(64-bit).
>ImportError: DLL load failed

### Trial failed on webUI
Please check the trial log file stderr for more details. If there is no such file and NNI is installed through pip, then you need to run PowerShell as administrator with this command first:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

If there is a stderr file, please check out. Two possible cases are as follows:

* forget to change the trial command `python3` into `python` in each experiment YAML.
* forget to install experiment dependencies such as TensorFlow, Keras and so on.

### Support tuner on Windows

* SMAC is not supported
* BOHB is supported, make sure C++ 14.0 compiler and dependencies installed successfully.

Note:

* If there is any error like `Segmentation fault`, please refer to [FAQ](FAQ.md)
