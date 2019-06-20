# Windows 上的 NNI（实验阶段的功能）

当前 Windows 上支持本机、远程和 OpenPAI 模式。 推荐 Windows 10 的 1809 版，其经过了测试。

## **在 Windows 上安装**

详细信息参考[安装文档](Installation.md)。

完成操作后，使用 **config_windows.yml** 配置来开始 Experiment 进行验证。

```bash
nnictl create --config nni\examples\trials\mnist\config_windows.yml
```

同样，其它示例的 YAML 配置中也需将 Trial 命令的 `python3` 替换为 `python`。

## **常见问答**

### 安装 NNI 时出现 simplejson 错误

确保安装了 C++ 14.0 编译器。

> building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified

### Trial failed with missing DLL in command line or PowerShell

This error caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and fail to install SciPy. Using Anaconda or Miniconda with Python(64-bit) can solve it.

> ImportError: DLL load failed

### Trial failed on webUI

Please check the trial log file stderr for more details.

If there is a stderr file, please check out. Two possible cases are as follows:

* 忘记将 Experiment 配置的 Trial 命令中的 `python3` 改为 `python`。
* 忘记安装 Experiment 的依赖，如 TensorFlow，Keras 等。

### Fail to use BOHB on Windows

Make sure C++ 14.0 compiler installed then try to run `nnictl package install --name=BOHB` to install the dependencies.

### Not supported tuner on Windows

SMAC is not supported currently, the specific reason can be referred to this [GitHub issue](https://github.com/automl/SMAC3/issues/483).

Note:

* 如果遇到如 `Segmentation fault` 这样的任何错误，参考[常见问题](FAQ.md)。