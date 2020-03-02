**在 DLTS 上运行 Experiment**
===
NNI 支持在 [DLTS](https://github.com/microsoft/DLWorkspace.git) 上运行 Experiment ，称之为 dlts 模式。 Before starting to use NNI dlts mode, you should have an account to access DLTS dashboard.

## Setup Environment

步骤 1. 从 DLTS 仪表板中选择集群，关于仪表板地址，需咨询管理员。

![选择集群](../../img/dlts-step1.png)

步骤 2. 准备 NNI 配置 YAML，如下所示：

```yaml
# 将此字段设置为 "dlts"
trainingServicePlatform: dlts
authorName: your_name
experimentName: auto_mnist
trialConcurrency: 2
maxExecDuration: 3h
maxTrialNum: 100
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 1
  image: msranni/nni
# 访问 DLTS 的配置
dltsConfig:
  dashboard: # Ask administrator for the cluster dashboard URL
```

Remember to fill the cluster dashboard URL to the last line.

步骤 3. Open your working directory of the cluster, paste the NNI config as well as related code to a directory.

![复制配置](../../img/dlts-step3.png)

步骤 4. Submit a NNI manager job to the specified cluster.

![提交 Job](../../img/dlts-step4.png)

步骤 5. Go to Endpoints tab of the newly created job, click the Port 40000 link to check trial's information.

![查看 NNI Web 界面](../../img/dlts-step5.png)
