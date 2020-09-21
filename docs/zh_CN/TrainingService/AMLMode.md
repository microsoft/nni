**在 Azure Machine Learning 上运行 Experiment**
===
NNI 支持在 [AML](https://azure.microsoft.com/zh-cn/services/machine-learning/) 上运行 Experiment，称为 aml 模式。

## 设置环境
步骤 1. 参考[指南](../Tutorial/QuickStart.md)安装 NNI。

步骤 2. 通过此 [链接](https://azure.microsoft.com/en-us/free/services/machine-learning/) 创建 Azure 账户/订阅。 如果已有 Azure 账户/订阅，跳过此步骤。

步骤 3. 在机器上安装 Azure CLI，参照[此](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)安装指南。

步骤 4. 从CLI验证您的Azure订阅。 要进行交互式身份验证，请打开命令行或终端并使用以下命令：
```
az login
```

步骤 5. 使用 Web 浏览器登录Azure帐户，并创建机器学习资源。 需要选择资源组并指定工作空间的名称。 之后下载 `config.json`，该文件将会在后面用到。 ![](../../img/aml_workspace.png)

步骤 6. 创建 AML 集群作为计算集群。 ![](../../img/aml_cluster.png)

步骤 7. 打开命令行并安装 AML 环境。
```
python3 -m pip install azureml
python3 -m pip install azureml-sdk
```

## 运行 Experiment
以 `examples/trials/mnist-tfv1` 为例。 NNI 的 YAML 配置文件如下：

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
trainingServicePlatform: aml
searchSpacePath: search_space.json
#可选项: true, false
useAnnotation: false
tuner:
  #可选项: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #可选项: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  image: msranni/nni
  gpuNum: 1
amlConfig:
  subscriptionId: ${replace_to_your_subscriptionId}
  resourceGroup: ${replace_to_your_resourceGroup}
  workspaceName: ${replace_to_your_workspaceName}
  computeTarget: ${replace_to_your_computeTarget}
```

注意：如果用 aml 模式运行，需要在 YAML 文件中设置 `trainingServicePlatform: aml`。

与[本机模式](LocalMode.md)的 Trial 配置相比，aml 模式下的键值还有：
* image
    * 必填。 作业中使用的 Docker 映像名称。 此示例中的镜像 `msranni/nni` 只支持 GPU 计算集群。

amlConfig:
* subscriptionId
    * 必填，Azure 订阅的 Id
* resourceGroup
    * 必填，Azure 订阅的资源组
* workspaceName
    * 必填，Azure 订阅的工作空间
* computeTarget
    * 必填，要在 AML 工作区中使用的计算机集群名称。 见步骤6。
* maxTrialNumPerGpu
    * 可选，用于指定 GPU 设备上的最大并发 Trial 的数量。
* useActiveGpu
    * 可选，用于指定 GPU 上存在其他进程时是否使用此 GPU。 默认情况下，NNI 仅在 GPU 中没有其他活动进程时才使用 GPU。

amlConfig 需要的信息可以从步骤 5 下载的 `config.json` 找到。

运行以下命令来启动示例示例 Experiment：
```
git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
cd nni/examples/trials/mnist-tfv1

# 修改 config.aml ...

nnictl create --config config_aml.yml
```
将 `${NNI_VERSION}` 替换为发布的版本或分支名称，例如：`v1.8`。
