# 在远程计算机上运行 Experiment

NNI 可以通过 SSH 在多个远程计算机上运行同一个 Experiment，称为 `remote` 模式。 这就像一个轻量级的训练平台。 在此模式下，可以从计算机启动 NNI，并将 Trial 并行调度到远程计算机。

## 远程计算机的要求

* 仅支持 Linux 作为远程计算机，其[配置需求](../Tutorial/Installation.md)与 NNI 本机模式相同。

* 根据[安装文章](../Tutorial/Installation.md)，在每台计算机上安装 NNI。

* 确保远程计算机满足 Trial 代码的环境要求。 如果默认环境不符合要求，可以将设置脚本添加到 NNI 配置的 `command` 字段。

* 确保远程计算机能被运行 `nnictl` 命令的计算机通过 SSH 访问。 同时支持 SSH 的密码和密钥验证方法。 有关高级用法，参考[配置](../Tutorial/ExperimentConfig.md)的 machineList 部分。

* 确保每台计算机上的 NNI 版本一致。

## 运行 Experiment

例如，有三台机器，可使用用户名和密码登录。

| IP       | 用户名 | 密码     |
| -------- | --- | ------ |
| 10.1.1.1 | bob | bob123 |
| 10.1.1.2 | bob | bob123 |
| 10.1.1.3 | bob | bob123 |

Install and run NNI on one of those three machines or another machine, which has network access to them.

Use `examples/trials/mnist-annotation` as the example. Below is content of `examples/trials/mnist-annotation/config_remote.yml`:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: remote
# 搜索空间文件
searchSpacePath: search_space.json
# 可选项: true, false
useAnnotation: true
tuner:
  # 可选项: TPE, Random, Anneal, Evolution, BatchTuner
  #SMAC (SMAC 需要先通过 nnictl 来安装)
  builtinTunerName: TPE
  classArgs:
    # 可选项:: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
#local 模式下 machineList 可为空
machineList:
  - ip: 10.1.1.1
    username: bob
    passwd: bob123
    #使用默认端口 22 时，该配置可跳过
    #port: 22
  - ip: 10.1.1.2
    username: bob
    passwd: bob123
  - ip: 10.1.1.3
    username: bob
    passwd: bob123
```

Files in `codeDir` will be uploaded to remote machines automatically. You can run below command on Windows, Linux, or macOS to spawn trials on remote Linux machines:

```bash
nnictl create --config examples/trials/mnist-annotation/config_remote.yml
```