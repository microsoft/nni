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

如果在实验运行时，升级了 nni 或删除了一些配置文件，会因为丢失配置文件而出现这类错误。 可以使用 `ps -ef | grep node` 命令来找到 Experiment 的 pid，并用 `kill -9 {pid}` 命令来停止 Experiment 进程。

### 无法在虚拟机的 NNI 网页中看到 `指标数据`

将虚拟机的网络配置为桥接模式来让虚拟机能被网络访问，并确保虚拟机的防火墙没有禁止相关端口。

### 帮助改进

在创建新问题前，请在 https://github.com/Microsoft/nni/issues 查看是否有人已经报告了相似的问题。