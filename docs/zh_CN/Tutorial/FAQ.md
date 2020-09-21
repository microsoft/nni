# 常见问答

此页为常见问题

### tmp 目录没空间了

nnictl 在执行时，使用 tmp 目录作为临时目录来复制 codeDir 下的文件。 当遇到下列错误时，先试试清空 **tmp** 目录。

> OSError: [Errno 28] No space left on device

### OpenPAI 模式下无法获得 Trial 的数据

在 OpenPAI 的训练模式下，NNI 管理器会在端口 51189 启动一个 RESTful 服务，来接收 OpenPAI 集群中 Trial 任务的指标数据。 如果在 OpenPAI 模式下的网页中不能看到任何指标，需要检查 51189 端口是否在防火墙规则中已打开。

### 安装时出现 Segmentation Fault (core dumped)

> make: *** [install-XXX] Segmentation fault (core dumped)

可依次试试以下方法：

* 更新或重新安装 Python 中的 pip： `python3 -m pip install -U pip`
* 在安装 NNI 时，添加 `--no-cache-dir` 参数：`python3 -m pip install nni --no-cache-dir`

### Job management error: getIPV4Address() failed because os.networkInterfaces().eth0 is undefined.

计算机没有 eth0 设备，需要在配置中设置 [nniManagerIp](ExperimentConfig.md)。

### 运行时间超过了 MaxDuration ，但没有停止

当 Experiment 到达最长运行时间时，nniManager 不会创建新的 Trial ，但除非手动停止 Experiment，运行中的 Trial 会继续运行直到结束。

### 使用 `nnictl stop` 无法停止 Experiment

如果在 Experiment 运行时，升级了 nni 或删除了一些配置文件，会因为丢失配置文件而出现这类错误。 可以使用 `ps -ef | grep node` 命令来找到 Experiment 的 PID，并用 `kill -9 {pid}` 命令来停止 Experiment 进程。

### 无法在虚拟机的 NNI 网页中看到 `指标数据`

将虚拟机的网络配置为桥接模式来让虚拟机能被网络访问，并确保虚拟机的防火墙没有禁止相关端口。

### 无法打开 Web 界面的链接

无法打开 Web 界面的链接可能有以下几个原因：

* `http://127.0.0.1`，`http://172.17.0.1` 以及 `http://10.0.0.15` 都是 localhost。如果在服务器或远程计算机上启动 Experiment， 可将此 IP 替换为所连接的 IP 来查看 Web 界面，如 `http://[远程连接的地址]:8080`
* 如果使用服务器 IP 后还是无法看到 Web 界面，可检查此服务器上是否有防火墙或需要代理。 或使用此运行 NNI Experiment 的服务器上的浏览器来查看 Web 界面。
* 另一个可能的原因是 Experiment 启动失败了，NNI 无法读取 Experiment 的信息。 可在如下目录中查看 NNIManager 的日志： `~/nni-experiments/[your_experiment_id]` `/log/nnimanager.log`

### RESTful 服务器启动失败

可能是网络配置有问题。可检查以下问题。

* 可能需要链接 `127.0.0.1` 与 `localhost`。 在 `/etc/hosts` 中增加 `127.0.0.1 localhost`。
* 也可能设置了一些代理。检查环境中是否有如 `HTTP_PROXY` 或 `HTTPS_PROXY` 的变量，如果有，则需要取消。

### NNI 在 Windows 上的问题

参考 [Windows 上的 NNI](InstallationWin.md)

### 更多常见问题解答

[标有常见问题标签的 Issue](https://github.com/microsoft/nni/labels/FAQ)

### 帮助改进

在创建新问题前，请在 https://github.com/Microsoft/nni/issues 查看是否有人已经报告了相似的问题。