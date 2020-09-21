# **在 OpenPAI 上运行 Experiment**

NNI 支持在 [OpenPAI](https://github.com/Microsoft/pai) 上运行 Experiment，即 pai 模式。 在使用 NNI 的 pai 模式前, 需要有 [OpenPAI](https://github.com/Microsoft/pai) 群集的账户。 如果没有 OpenPAI 账户，参考[这里](https://github.com/Microsoft/pai#how-to-deploy)来进行部署。 在 pai 模式中，会在 Docker 创建的容器中运行 Trial 程序。

## 设置环境

步骤 1. 参考[指南](../Tutorial/QuickStart.md)安装 NNI。

步骤 2. 获得令牌（token）。

打开 OpenPAI 的 Web 界面，并点击右上方的 `My profile` 按钮。 ![](../../img/pai_profile.jpg)

点击页面中的 `copy` 按钮来复制 jwt 令牌。 ![](../../img/pai_token.jpg)

步骤 3. 将 NFS 存储挂在到本地计算机。

点击 Web 界面中的 `Submit job` 按钮。 ![](../../img/pai_job_submission_page.jpg)

找到作业提交页面中的数据管理部分。 ![](../../img/pai_data_management_page.jpg)

`Preview container paths` 是 API 提供的 NFS 主机和路径，需要将对应的位置挂载到本机，然后 NNI 才能使用 NFS 存储。  
例如，使用下列命令：

```bash
sudo mount -t nfs4 gcr-openpai-infra02:/pai/data /local/mnt
```

然后容器中的 `/data` 路径会被挂载到本机的 `/local/mnt` 文件夹  
然后在 NNI 的配置文件中如下配置：

```yaml
nniManagerNFSMountPath: /local/mnt
```

步骤 4. 获得 OpenPAI 存储的配置名称和 nniManagerMountPath

`Team share storage` 字段是在 OpenPAI 中指定存储配置的值。 可以在 `Team share storage` 中找到 `paiStorageConfigName` 和 `containerNFSMountPath` 字段，如：

```yaml
paiStorageConfigName: confignfs-data
containerNFSMountPath: /mnt/confignfs-data
```

## 运行 Experiment

以 `examples/trials/mnist-annotation` 为例。 NNI 的 YAML 配置文件如下：

```yaml
authorName: your_name
experimentName: auto_mnist
# 并发运行的 Trial 数量
trialConcurrency: 2
# Experiment 的最长持续运行时间
maxExecDuration: 3h
# 空表示一直运行
maxTrialNum: 100
# 可选项: local, remote, pai
trainingServicePlatform: pai
# 搜索空间文件
searchSpacePath: search_space.json
# 可选项: true, false
useAnnotation: true
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: ~/nni/examples/trials/mnist-annotation
  gpuNum: 0
  cpuNum: 1
  memoryMB: 8196
  image: msranni/nni:latest
  virtualCluster: default
  nniManagerNFSMountPath: /local/mnt
  containerNFSMountPath: /mnt/confignfs-data
  paiStorageConfigName: confignfs-data
# 配置要访问的 OpenPAI 集群
paiConfig:
  userName: your_pai_nni_user
  token: your_pai_token
  host: 10.1.1.1
  # 可选，测试版功能。
  reuse: true
```

注意：如果用 pai 模式运行，需要在 YAML 文件中设置 `trainingServicePlatform: pai`。 配置文件中的 host 字段是 OpenPAI 作业提交页面的地址，例如：`10.10.5.1`，NNI 中默认协议是 `http`，如果 OpenPAI 集群启用了 https，则需要使用 `https://10.10.5.1` 的格式。

### Trial 配置

与[本机模式](LocalMode.md)，以及[远程计算机模式](RemoteMachineMode.md)相比，pai 模式的 `trial` 需要额外的配置：

* cpuNum
    
    可选。 Trial 程序的 CPU 需求，必须为正数。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* memoryMB
    
    可选。 Trial 程序的内存需求，必须为正数。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* image
    
    可选。 在 pai 模式中，Trial 程序由 OpenPAI 在 [Docker 容器](https://www.docker.com/)中安排运行。 此字段用来指定 Trial 程序的容器使用的 Docker 映像。
    
    [Docker Hub](https://hub.docker.com/) 上有预制的 NNI Docker 映像 [nnimsra/nni](https://hub.docker.com/r/msranni/nni/)。 它包含了用来启动 NNI Experiment 所依赖的所有 Python 包，Node 模块和 JavaScript。 生成此 Docker 映像的文件在[这里](https://github.com/Microsoft/nni/tree/master/deployment/docker/Dockerfile)。 可以直接使用此映像，或参考它来生成自己的映像。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* virtualCluster
    
    可选。 设置 OpenPAI 的 virtualCluster，即虚拟集群。 如果未设置此参数，将使用默认（default）虚拟集群。

* nniManagerNFSMountPath
    
    必填。 在 nniManager 计算机上设置挂载的路径。

* containerNFSMountPath
    
    必填。 在 OpenPAI 的容器中设置挂载路径。

* paiStorageConfigName:
    
    可选。 设置 OpenPAI 中使用的存储名称。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* command
    
    可选。 设置 OpenPAI 容器中使用的命令。

* paiConfigPath 可选。 设置 OpenPAI 作业配置文件路径，文件为 YAML 格式。
    
    如果在 NNI 配置文件中设置了 `paiConfigPath`，则不需在 `trial` 配置中设置 `command`, `paiStorageConfigName`, `virtualCluster`, `image`, `memoryMB`, `cpuNum`, `gpuNum`。 这些字段将使用 `paiConfigPath` 指定的配置文件中的值。
    
    注意：
    
    1. OpenPAI 配置文件中的作业名称会由 NNI 指定，格式为：nni_exp_${this.experimentId}*trial*${trialJobId}。
    
    2. 如果在 OpenPAI 配置文件中有多个 taskRoles，NNI 会将这些 taksRoles 作为一个 Trial 任务，用户需要确保只有一个 taskRole 会将指标上传到 NNI 中，否则可能会产生错误。

### OpenPAI 配置

`paiConfig` 包括了 OpenPAI 的专门配置，

* userName
    
    必填。 OpenPAI 平台的用户名。

* token
    
    必填。 OpenPAI 平台的身份验证密钥。

* host
    
    必填。 OpenPAI 平台的主机。 OpenPAI 作业提交页面的地址，例如：`10.10.5.1`，NNI 中默认协议是 `http`，如果 OpenPAI 集群启用了 https，则需要使用 `https://10.10.5.1` 的格式。

* reuse (测试版功能)
    
    可选，默认为 false。 如果为 true，NNI 会重用 OpenPAI 作业，在其中运行尽可能多的 Trial。 这样可以节省创建新作业的时间。 用户需要确保同一作业中的每个 Trial 相互独立，例如，要避免从之前的 Trial 中读取检查点。

完成并保存 NNI Experiment 配置文件后（例如可保存为：exp_pai.yml），运行以下命令：

```bash
nnictl create --config exp_pai.yml
```

来在 pai 模式下启动 Experiment。 NNI 会为每个 Trial 创建 OpenPAI 作业，作业名称的格式为 `nni_exp_{experiment_id}_trial_{trial_id}`。 可以在 OpenPAI 集群的网站中看到 NNI 创建的作业，例如： ![](../../img/nni_pai_joblist.jpg)

注意：pai 模式下，NNIManager 会启动 RESTful 服务，监听端口为 NNI 网页服务器的端口加1。 例如，如果网页端口为`8080`，那么 RESTful 服务器会监听在 `8081`端口，来接收运行在 Kubernetes 中的 Trial 作业的指标。 因此，需要在防火墙中启用端口 `8081` 的 TCP 协议，以允许传入流量。

当一个 Trial 作业完成后，可以在 NNI 网页的概述页面（如：http://localhost:8080/oview）中查看 Trial 的信息。

在 Trial 列表页面中展开 Trial 信息，点击如下的 logPath： ![](../../img/nni_webui_joblist.jpg)

接着将会打开 HDFS 的 WEB 界面，并浏览到 Trial 的输出文件： ![](../../img/nni_trial_hdfs_output.jpg)

在输出目录中可以看到三个文件：stderr, stdout, 以及 trial.log

## 数据管理

使用 NNI 启动 Experiment 前，应在 nniManager 计算机中设置相应的挂载数据的路径。 OpenPAI 有自己的存储（NFS、AzureBlob ...），在 OpenPAI 中使用的存储将在启动作业时挂载到容器中。 应通过 `paiStorageConfigName` 字段选择 OpenPAI 中的存储类型。 然后，应将存储挂载到 nniManager 计算机上，并在配置文件中设置 `nniManagerNFSMountPath`，NNI会生成 bash 文件并将 `codeDir` 中的数据拷贝到 `nniManagerNFSMountPath` 文件夹中，然后启动 Trial 任务。 `nniManagerNFSMountPath` 中的数据会同步到 OpenPAI 存储中，并挂载到 OpenPAI 的容器中。 容器中的数据路径在 `containerNFSMountPath` 设置，NNI 将进入该文件夹，运行脚本启动 Trial 任务。

## 版本校验

从 0.6 开始，NNI 支持版本校验。确保 NNIManager 与 trialKeeper 的版本一致，避免兼容性错误。 检查策略：

1. 0.6 以前的 NNIManager 可与任何版本的 trialKeeper 一起运行，trialKeeper 支持向后兼容。
2. 从 NNIManager 0.6 开始，与 triakKeeper 的版本必须一致。 例如，如果 NNIManager 是 0.6 版，则 trialKeeper 也必须是 0.6 版。
3. 注意，只有版本的前两位数字才会被检查。例如，NNIManager 0.6.1 可以和 trialKeeper 的 0.6 或 0.6.2 一起使用，但不能与 trialKeeper 的 0.5.1 或 0.7 版本一起使用。

如果 Experiment 无法运行，而且不能确认是否是因为版本不匹配造成的，可以在 Web 界面检查是否有相关的错误消息。 ![](../../img/version_check.png)