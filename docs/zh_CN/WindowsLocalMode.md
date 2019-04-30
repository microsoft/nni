# Windows 本地模式（测试中）

当前 Windows 下仅支持本机模式。 推荐 Windows 10 的 1809 版，其经过了测试。

## **在 Windows 上安装**

**强烈推荐使用 Anaconda python(64 位)。**

在第一次使用 PowerShell 运行脚本时，需要用**使用管理员权限**运行如下命令：

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

When these things are done, run the **config_windows.yml** file from your command line to start the experiment.

```bash
    nnictl create --config nni/examples/trials/mnist/config_windows.yml
```

同样，其它示例的 YAML 配置中也需将 Trial 命令的 `python3` 替换为 `python`。

## **Frequent met errors and answers**

### 安装 NNI 时出现 simplejson 错误

确保安装了 C++ 14.0 编译器。

> building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### 从源代码安装 NNI 时，遇到 PowerShell 错误

如果第一次运行 PowerShell 脚本，且没有设置过执行脚本的策略，会遇到下列错误。 需要以管理员身份运行此命令：

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

> ...cannot be loaded because running scripts is disabled on this system.

### 在命令行或 PowerShell 中，Trial 因为缺少 DLL 而失败

此错误因为缺少 LIBIFCOREMD.DLL 和 LIBMMD.DLL 文件，且 SciPy 安装失败。 强烈推荐使用 Anaconda python。 如果要使用官方的 Python，确保运行 NNI 前，在 `Visual Studio`，`MATLAB`，`MKL` 和`Intel Distribution for Python` 中至少安装了一个。 如果没有，则需要安装其中之一，或使用 Anaconda Python （64位）。

> ImportError: DLL load failed

### Web 界面上的 Trial 错误

Please check the trial log file stderr for more details. If there is no such file and nni is installed through pip, then you need to run powershell as administrator with this command first:

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

If there is a stderr file, please check out. Two possible cases are as follows:

* forget to change the trial command `python3` into `python` in each experiment yaml.
* forget to install experiment dependencies such as tensorflow, keras and so on.

### Support tuner on Windows

* SMAC is not supported
* BOHB is supported, make sure C++ 14.0 compiler and dependencies installed successfully.

Note:

* If there is any error like `Segmentation fault`, please refer to [FAQ](FAQ.md)