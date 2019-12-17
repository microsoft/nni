# 在多机上运行 Experiment

NNI 支持通过 SSH 通道在多台计算机上运行 Experiment，称为 `remote` 模式。 NNI 需要这些计算机的访问权限，并假定已配置好了深度学习训练环境。

例如：有三台服务器，登录账户为 `bob`（注意：账户不必在各台计算机上一致）：

| IP       | 用户名 | 密码     |
| -------- | --- | ------ |
| 10.1.1.1 | bob | bob123 |
| 10.1.1.2 | bob | bob123 |
| 10.1.1.3 | bob | bob123 |

## 设置 NNI 环境

按照[指南](../Tutorial/QuickStart.md)在每台计算机上安装 NNI。

## 运行 Experiment

将 NNI 安装在可以访问上述三台计算机的网络的另一台计算机上，或者仅在三台计算机中的任何一台上运行 `nnictl` 即可启动 Experiment。

以 `examples/trials/mnist-annotation` 为例。 此处示例在 `examples/trials/mnist-annotation/config_remote.yml`：

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

`codeDir` 中的文件会被自动上传到远程服务器。 可在不同的操作系统上运行 NNI (Windows, Linux, MacOS)，来在远程机器上（仅支持 Linux）运行 Experiment。

```bash
nnictl create --config examples/trials/mnist-annotation/config_remote.yml
```

也可使用公钥/私钥对，而非用户名/密码进行身份验证。 有关高级用法，请参考[实验配置参考](../Tutorial/ExperimentConfig.md)。

## 版本校验

从 0.6 开始，NNI 支持版本校验，详情参考[这里](PaiMode.md)。