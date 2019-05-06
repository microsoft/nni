# Windows 本地模式（测试中）

当前 Windows 下仅支持本机模式。 推荐 Windows 10 的 1809 版，其经过了测试。

## **在 Windows 上安装**

**强烈推荐使用 Anaconda python(64 位)。**

When you use PowerShell to run script for the first time, you need **run PowerShell as administrator** with this command:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

* **通过 pip 命令安装 NNI**
    
    先决条件：`python(64-bit) >= 3.5`

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

When these things are done, use the **config_windows.yml** configuration to start an experiment for validation.

```bash
nnictl create --config nni/examples/trials/mnist/config_windows.yml
```

For other examples you need to change trial command `python3` into `python` in each example YAML.

## **FAQ**

### simplejson failed when installing NNI

确保安装了 C++ 14.0 编译器。

> building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### Fail to run PowerShell when install NNI from source

If you run PowerShell script for the first time and did not set the execution policies for executing the script, you will meet this error below. Try to run PowerShell as administrator with this command first:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

> ...cannot be loaded because running scripts is disabled on this system.

### Trial failed with missing DLL in cmd or PowerShell

This error caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and fail to install SciPy. 强烈推荐使用 Anaconda python。 If you use official python, make sure you have one of `Visual Studio`, `MATLAB`, `MKL` and `Intel Distribution for Python` installed on Windows before running NNI. If not, try to install one of products above or Anaconda python(64-bit).

> ImportError: DLL load failed

### Web 界面上的 Trial 错误

检查 Trial 日志文件来了解详情。 If there is no such file and NNI is installed through pip, then you need to run PowerShell as administrator with this command first:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

如果存在 stderr 文件，也需要查看其内容。 可能的错误情况包括：

* forget to change the trial command `python3` into `python` in each experiment YAML.
* forget to install experiment dependencies such as TensorFlow, Keras and so on.

### Windows 上支持的 Tuner

* 不支持 SMAC
* 支持 BOHB，但需要确保安装了 C++ 14.0 编译器。

注意：

* 如果遇到 `Segmentation fault` 这样的错误，参考[常见问答](FAQ.md)。