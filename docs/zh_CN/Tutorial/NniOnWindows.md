# Windows 上的 NNI（实验阶段的功能）

在 Windows 上运行 NNI 是测试中的功能。 推荐 Windows 10 的 1809 版，其经过了测试。

## **在 Windows 上安装**

详细信息参考[安装文档](Installation.md)。

完成操作后，使用 **config_windows.yml** 配置来开始 Experiment 进行验证。

```bash
nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
```

同样，其它示例的 YAML 配置中也需将 Trial 命令的 `python3` 替换为 `python`。

## **常见问答**

### 安装 NNI 时出现 simplejson 错误

确保安装了 C++ 14.0 编译器。

> building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### 在命令行或 PowerShell 中，Trial 因为缺少 DLL 而失败

此错误因为缺少 LIBIFCOREMD.DLL 和 LIBMMD.DLL 文件，且 SciPy 安装失败。 使用 Anaconda 或 Miniconda 和 Python（64位）可解决。

> ImportError: DLL load failed

### Web 界面上的 Trial 错误

检查 Trial 日志文件来了解详情。

如果存在 stderr 文件，也需要查看其内容。 可能的错误情况包括：

* 忘记将 Experiment 配置的 Trial 命令中的 `python3` 改为 `python`。
* 忘记安装 Experiment 的依赖，如 TensorFlow，Keras 等。

### 无法在 Windows 上使用 BOHB

确保安装了 C ++ 14.0 编译器然后尝试运行 `nnictl package install --name=BOHB` 来安装依赖项。

### Windows 上不支持的 Tuner

当前不支持 SMAC，原因可参考[此问题](https://github.com/automl/SMAC3/issues/483)。

### 将 Windows 服务器用作远程服务器

目前不支持。

注意：

* 如果遇到如 `Segmentation fault` 这样的任何错误，参考[常见问题](FAQ.md)。