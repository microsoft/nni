# FAQ

此页为常见问题

### tmp 目录没空间了

nnictl 在执行时，使用 tmp 目录作为临时目录来复制 codeDir 下的文件。 当遇到下列错误时，先试试清空 **tmp** 目录。

> OSError: [Errno 28] No space left on device

### OpenPAI 模式下无法获得 Trial 的数据

In OpenPAI training mode, we start a rest server which listens on 51189 port in NNI Manager to receive metrcis reported from trials running in OpenPAI cluster. If you didn't see any metrics from WebUI in OpenPAI mode, check your machine where NNI manager runs on to make sure 51189 port is turned on in the firewall rule.

### 安装时出现 Segmentation Fault (core dumped)

> make: *** [install-XXX] Segmentation fault (core dumped)

可依次试试以下方法：

* 更新或重新安装 Python 中的 pip： `python3 -m pip install -U pip`
* Install NNI with `--no-cache-dir` flag like `python3 -m pip install nni --no-cache-dir`

### 作业管理错误：getIPV4Address() failed because os.networkInterfaces().eth0 is undefined.

计算机没有 eth0 设备，需要在配置文件中手动设置 nniManagerIp。 [参考此处](https://github.com/Microsoft/nni/blob/master/docs/ExperimentConfig.md)

### 运行时间超过了 MaxDuration ，但没有停止

当 Experiment 到达最长运行时间时，nniManager 不会创建新的 Trial ，但除非手动停止 Experiment，运行中的 Trial 会继续运行直到结束。

### 使用 `nnictl stop` 无法停止 Experiment

If you upgrade your NNI or you delete some config files of NNI when there is an experiment running, this kind of issue may happen because the loss of config file. 可以使用 `ps -ef | grep node` 命令来找到 Experiment 的 pid，并用 `kill -9 {pid}` 命令来停止 Experiment 进程。

### 无法在虚拟机的 NNI 网页中看到 `指标数据`

将虚拟机的网络配置为桥接模式来让虚拟机能被网络访问，并确保虚拟机的防火墙没有禁止相关端口。

### 帮助改进

在创建新问题前，请在 https://github.com/Microsoft/nni/issues 查看是否有人已经报告了相似的问题。