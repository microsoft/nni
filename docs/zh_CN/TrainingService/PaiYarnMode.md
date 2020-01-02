**在 OpenPAIYarn 上运行 Experiment**
===
原始的 `pai` 模式改为了 `paiYarn` 模式，这是基于 Yarn 的分布式训练平台。

## 设置环境
参考[指南](../Tutorial/QuickStart.md)安装 NNI。

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
# 可选项: local, remote, pai, paiYarn
trainingServicePlatform: paiYarn
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
# 配置访问的 OpenpaiYarn 集群
paiYarnConfig:
  userName: your_paiYarn_nni_user
  passWord: your_paiYarn_password
  host: 10.1.1.1
```

注意：如果用 paiYarn 模式运行，需要在 YAML 文件中设置 `trainingServicePlatform: paiYarn`。

与[本机模式](LocalMode.md)，以及[远程计算机模式](RemoteMachineMode.md)相比，paiYarn 模式的 Trial 有额外的配置：
* cpuNum
    * 必填。 Trial 程序的 CPU 需求，必须为正数。
* memoryMB
    * 必填。 Trial 程序的内存需求，必须为正数。
* image
    * 必填。 在 paiYarn 模式中，Trial 程序由 OpenpaiYarn 在 [Docker 容器](https://www.docker.com/)中安排运行。 此键用来指定 Trial 程序的容器使用的 Docker 映像。
    * [Docker Hub](https://hub.docker.com/) 上有预制的 NNI Docker 映像 [nnimsra/nni](https://hub.docker.com/r/msranni/nni/)。 它包含了用来启动 NNI Experiment 所依赖的所有 Python 包，Node 模块和 JavaScript。 生成此 Docker 映像的文件在[这里](https://github.com/Microsoft/nni/tree/master/deployment/docker/Dockerfile)。 可以直接使用此映像，或参考它来生成自己的映像。
* virtualCluster
    * 可选。 设置 OpenPAIYarn 的 virtualCluster，即虚拟集群。 如果未设置此参数，将使用默认（default）虚拟集群。
* shmMB
    * 可选。 设置 OpenPAIYarn 的 shmMB，即 Docker 中的共享内存。
* authFile
    * 可选。在使用 paiYarn 模式时，为私有 Docker 仓库设置认证文件，[见参考文档](https://github.com/microsoft/paiYarn/blob/2ea69b45faa018662bc164ed7733f6fdbb4c42b3/docs/faq.md#q-how-to-use-private-docker-registry-job-image-when-submitting-an-openpaiYarn-job)。提供 authFile 的本地路径即可， NNI 会上传此文件。
* portList
    * 可选。 Set the portList configuration of OpenpaiYarn, it specifies a list of port used in container, [Refer](https://github.com/microsoft/paiYarn/blob/b2324866d0280a2d22958717ea6025740f71b9f0/docs/job_tutorial.md#specification). The config schema in NNI is shown below:
    ```
    portList:
      - label: test
        beginAt: 8080
        portNumber: 2
    ```
    Let's say you want to launch a tensorboard in the mnist example using the port. So the first step is to write a wrapper script `launch_paiYarn.sh` of `mnist.py`.

    ```bash
    export TENSORBOARD_PORT=paiYarn_PORT_LIST_${paiYarn_CURRENT_TASK_ROLE_NAME}_0_tensorboard
    tensorboard --logdir . --port ${!TENSORBOARD_PORT} &
    python3 mnist.py
    ```
    The config file of portList should be filled as following:

    ```yaml
  trial:
    command: bash launch_paiYarn.sh
    portList:
      - label: tensorboard
        beginAt: 0
        portNumber: 1
    ```

NNI support two kind of authorization method in paiYarn, including password and paiYarn token, [refer](https://github.com/microsoft/paiYarn/blob/b6bd2ab1c8890f91b7ac5859743274d2aa923c22/docs/rest-server/API.md#2-authentication). The authorization is configured in `paiYarnConfig` field. For password authorization, the `paiYarnConfig` schema is:
```
paiYarnConfig:
  userName: your_paiYarn_nni_user
  passWord: your_paiYarn_password
  host: 10.1.1.1
```
For paiYarn token authorization, the `paiYarnConfig` schema is:
```
paiYarnConfig:
  userName: your_paiYarn_nni_user
  token: your_paiYarn_token
  host: 10.1.1.1
```

Once complete to fill NNI experiment config file and save (for example, save as exp_paiYarn.yml), then run the following command
```
nnictl create --config exp_paiYarn.yml
```
to start the experiment in paiYarn mode. NNI will create OpenpaiYarn job for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. You can see jobs created by NNI in the OpenpaiYarn cluster's web portal, like: ![](../../img/nni_paiYarn_joblist.jpg)

Notice: In paiYarn mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic.

Once a trial job is completed, you can goto NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information.

Expand a trial information in trial list view, click the logPath link like: ![](../../img/nni_webui_joblist.jpg)

And you will be redirected to HDFS web portal to browse the output files of that trial in HDFS: ![](../../img/nni_trial_hdfs_output.jpg)

You can see there're three fils in output folder: stderr, stdout, and trial.log

## data management
If your training data is not too large, it could be put into codeDir, and nni will upload the data to hdfs, or you could build your own docker image with the data. If you have large dataset, it's not appropriate to put the data in codeDir, and you could follow the [guidance](https://github.com/microsoft/paiYarn/blob/master/docs/user/storage.md) to mount the data folder in container.

If you also want to save trial's other output into HDFS, like model files, you can use environment variable `NNI_OUTPUT_DIR` in your trial code to save your own output files, and NNI SDK will copy all the files in `NNI_OUTPUT_DIR` from trial's container to HDFS, the target path is `hdfs://host:port/{username}/nni/{experiments}/{experimentId}/trials/{trialId}/nnioutput`

## version check
NNI support version check feature in since version 0.6. It is a policy to insure the version of NNIManager is consistent with trialKeeper, and avoid errors caused by version incompatibility. Check policy:
1. NNIManager before v0.6 could run any version of trialKeeper, trialKeeper support backward compatibility.
2. Since version 0.6, NNIManager version should keep same with triakKeeper version. For example, if NNIManager version is 0.6, trialKeeper version should be 0.6 too.
3. Note that the version check feature only check first two digits of version.For example, NNIManager v0.6.1 could use trialKeeper v0.6 or trialKeeper v0.6.2, but could not use trialKeeper v0.5.1 or trialKeeper v0.7.

If you could not run your experiment and want to know if it is caused by version check, you could check your webUI, and there will be an error message about version check. ![](../../img/version_check.png)
