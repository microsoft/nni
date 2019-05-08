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

* __通过代码安装 NNI__

  先决条件: `python >=3.5`, `git`, `powershell`

  ```bash
  git clone -b v0.7 https://github.com/Microsoft/nni.git
  cd nni
  powershell ./install.ps1
  ```

运行完以上脚本后，从命令行使用 **config_windows.yml** 来启动 Experiment，完成安装验证。

```bash
nnictl create --config nni/examples/trials/mnist/config_windows.yml
```

同样，其它示例的 YAML 配置中也需将 Trial 命令的 `python3` 替换为 `python`。

## **常见问答**

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

此错误因为缺少 LIBIFCOREMD.DLL 和 LIBMMD.DLL 文件，且 SciPy 安装失败。 使用 Anaconda Python(64-bit) 可解决此问题。

> ImportError: DLL load failed

### Web 界面上的 Trial 错误

检查 Trial 日志文件来了解详情。 如果没有日志文件，且 NNI 是通过 pip 安装的，则需要在管理员权限下先运行以下命令：

```bash
Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

如果存在 stderr 文件，也需要查看其内容。 可能的错误情况包括：

* 忘记将 Experiment 配置的 Trial 命令中的 `python3` 改为 `python`。
* 忘记安装 Experiment 的依赖，如 TensorFlow，Keras 等。

### 无法在 Windows 上使用 BOHB

确保安装了 C ++ 14.0 编译器然后尝试运行 `nnictl package install --name=BOHB` 来安装依赖项。

### Windows 上不支持的 Tuner

当前不支持 SMAC，原因可参考[此问题](https://github.com/automl/SMAC3/issues/483)。

Note:

* If there is any error like `Segmentation fault`, please refer to [FAQ](FAQ.md)