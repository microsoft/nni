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
  nniManagerNFSMountPath: /home/user/mnt
  containerNFSMountPath: /mnt/data/user
  paiStorageConfigName: confignfs-data
# 配置要访问的 OpenPAI 集群
paiConfig:
  userName: your_pai_nni_user
  token: your_pai_token
  host: 10.1.1.1
  # 可选，测试版功能。
  reuse: true
```

注意：如果用 pai 模式运行，需要在 YAML 文件中设置 `trainingServicePlatform: pai`。

### Trial configurations

Compared with [LocalMode](LocalMode.md) and [RemoteMachineMode](RemoteMachineMode.md), `trial` configuration in pai mode have these additional keys:

* cpuNum
    
    Optional key. Should be positive number based on your trial program's CPU requirement. If it is not set in trial configuration, it should be set in the config file specified in `paiConfigPath` field.

* memoryMB
    
    Optional key. Should be positive number based on your trial program's memory requirement. If it is not set in trial configuration, it should be set in the config file specified in `paiConfigPath` field.

* image
    
    Optional key. In pai mode, your trial program will be scheduled by OpenPAI to run in [Docker container](https://www.docker.com/). This key is used to specify the Docker image used to create the container in which your trial will run.
    
    We already build a docker image [nnimsra/nni](https://hub.docker.com/r/msranni/nni/) on [Docker Hub](https://hub.docker.com/). It contains NNI python packages, Node modules and javascript artifact files required to start experiment, and all of NNI dependencies. The docker file used to build this image can be found at [here](https://github.com/Microsoft/nni/tree/master/deployment/docker/Dockerfile). You can either use this image directly in your config file, or build your own image based on it. If it is not set in trial configuration, it should be set in the config file specified in `paiConfigPath` field.

* virtualCluster
    
    Optional key. Set the virtualCluster of OpenPAI. If omitted, the job will run on default virtual cluster.

* nniManagerNFSMountPath
    
    Required key. Set the mount path in your nniManager machine.

* containerNFSMountPath
    
    Required key. Set the mount path in your container used in OpenPAI.

* paiStorageConfigName:
    
    Optional key. Set the storage name used in OpenPAI. If it is not set in trial configuration, it should be set in the config file specified in `paiConfigPath` field.

* command
    
    Optional key. Set the commands used in OpenPAI container.

* paiConfigPath Optional key. Set the file path of OpenPAI job configuration, the file is in yaml format.
    
    If users set `paiConfigPath` in NNI's configuration file, no need to specify the fields `command`, `paiStorageConfigName`, `virtualCluster`, `image`, `memoryMB`, `cpuNum`, `gpuNum` in `trial` configuration. These fields will use the values from the config file specified by `paiConfigPath`.
    
    Note:
    
    1. The job name in OpenPAI's configuration file will be replaced by a new job name, the new job name is created by NNI, the name format is nni_exp_${this.experimentId}*trial*${trialJobId}.
    
    2. If users set multiple taskRoles in OpenPAI's configuration file, NNI will wrap all of these taksRoles and start multiple tasks in one trial job, users should ensure that only one taskRole report metric to NNI, otherwise there might be some conflict error.

### OpenPAI configurations

`paiConfig` includes OpenPAI specific configurations,

* userName
    
    Required key. User name of OpenPAI platform.

* token
    
    Required key. Authentication key of OpenPAI platform.

* host
    
    Required key. The host of OpenPAI platform. It's OpenPAI's job submission page uri, like `10.10.5.1`, the default http protocol in NNI is `http`, if your OpenPAI cluster enabled https, please use the uri in `https://10.10.5.1` format.

* reuse (experimental feature)
    
    Optional key, default is false. If it's true, NNI will reuse OpenPAI jobs to run as many as possible trials. It can save time of creating new jobs. User needs to make sure each trial can run independent in same job, for example, avoid loading checkpoint from previous trials.

Once complete to fill NNI experiment config file and save (for example, save as exp_pai.yml), then run the following command

```bash
nnictl create --config exp_pai.yml
```

to start the experiment in pai mode. NNI will create OpenPAI job for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. You can see jobs created by NNI in the OpenPAI cluster's web portal, like: ![](../../img/nni_pai_joblist.jpg)

Notice: In pai mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic.

Once a trial job is completed, you can goto NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information.

Expand a trial information in trial list view, click the logPath link like: ![](../../img/nni_webui_joblist.jpg)

And you will be redirected to HDFS web portal to browse the output files of that trial in HDFS: ![](../../img/nni_trial_hdfs_output.jpg)

You can see there're three fils in output folder: stderr, stdout, and trial.log

## 数据管理

Before using NNI to start your experiment, users should set the corresponding mount data path in your nniManager machine. OpenPAI has their own storage(NFS, AzureBlob ...), and the storage will used in OpenPAI will be mounted to the container when it start a job. Users should set the OpenPAI storage type by `paiStorageConfigName` field to choose a storage in OpenPAI. Then users should mount the storage to their nniManager machine, and set the `nniManagerNFSMountPath` field in configuration file, NNI will generate bash files and copy data in `codeDir` to the `nniManagerNFSMountPath` folder, then NNI will start a trial job. The data in `nniManagerNFSMountPath` will be sync to OpenPAI storage, and will be mounted to OpenPAI's container. The data path in container is set in `containerNFSMountPath`, NNI will enter this folder first, and then run scripts to start a trial job.

## 版本校验

NNI support version check feature in since version 0.6. It is a policy to insure the version of NNIManager is consistent with trialKeeper, and avoid errors caused by version incompatibility. Check policy:

1. 0.6 以前的 NNIManager 可与任何版本的 trialKeeper 一起运行，trialKeeper 支持向后兼容。
2. 从 NNIManager 0.6 开始，与 triakKeeper 的版本必须一致。 例如，如果 NNIManager 是 0.6 版，则 trialKeeper 也必须是 0.6 版。
3. 注意，只有版本的前两位数字才会被检查。例如，NNIManager 0.6.1 可以和 trialKeeper 的 0.6 或 0.6.2 一起使用，但不能与 trialKeeper 的 0.5.1 或 0.7 版本一起使用。

If you could not run your experiment and want to know if it is caused by version check, you could check your webUI, and there will be an error message about version check. ![](../../img/version_check.png)