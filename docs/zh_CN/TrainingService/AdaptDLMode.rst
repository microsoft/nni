在 AdaptDL 上运行 Experiment
============================

NNI 支持在 `AdaptDL <https://github.com/petuum/adaptdl>`__ 上运行，称为 AdaptDL 模式。 在开始使用 NNI 的 AdaptDL 模式前，需要有一个 Kubernetes 集群，可以是私有部署的，或者是 `Azure Kubernetes Service(AKS) <https://azure.microsoft.com/zh-cn/services/kubernetes-service/>`__，并需要一台配置好  `kubeconfig <https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/>`__ 的 Ubuntu 计算机连接到此 Kubernetes 集群。 在 AdaptDL 模式下，每个 Trial 程序会在 AdaptDL 集群中作为一个 Kubeflow 作业来运行。

AdaptDL 旨在使动态资源环境（例如共享集群和云）中的分布式深度学习变得轻松高效。

部署 Kubernetes 的准备工作
-----------------------------------


#. 采用 **Kubernetes 1.14** 或更高版本。 根据下面的指南设置 Kubernetes 环境： `on Azure <https://azure.microsoft.com/zh-cn/services/kubernetes-service/>`__\ ， `on-premise <https://kubernetes.io/docs/setup/>`__ ， `cephfs <https://kubernetes.io/docs/concepts/storage/storage-classes/#ceph-rbd>`__\  和  `microk8s with storage add-on enabled <https://microk8s.io/docs/addons>`__。
#. Helm 将 **AdaptDL Scheduler** 安装到 Kubernetes 集群中。 参照 `指南 <https://adaptdl.readthedocs.io/en/latest/installation/install-adaptdl.html>`__ 来设置 AdaptDL scheduler。
#. 配置 **kubeconfig** 文件，NNI 将使用此配置与 Kubernetes API 服务交互。 默认情况下，NNI 管理器会使用 $(HOME)/.kube/config 作为 kubeconfig 文件的路径。 也可以通过环境变量 **KUBECONFIG** 来指定其它 kubeconfig 文件。 根据 `指南 <https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig>`__ 了解更多 kubeconfig 的信息。
#. 如果 NNI Trial 作业需要 GPU 资源，需按照 `指南 <https://github.com/NVIDIA/k8s-device-plugin>`__ 来配置 **Kubernetes 下的 Nvidia 插件**。
#. （可选）准备 **NFS服务器** 并导出通用装载作为外部存储。
#. 参考 `指南 <../Tutorial/QuickStart.rst>`__ 安装 **NNI**。

验证先决条件
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   nnictl --version
   # Expected: <version_number>

.. code-block:: bash

   kubectl version
   # Expected that the kubectl client version matches the server version.

.. code-block:: bash

   kubectl api-versions | grep adaptdl
   # Expected: adaptdl.petuum.com/v1

运行实验
-----------------

在 ``examples/trials/cifar10_pytorch`` 目录下，``CIFAR10`` 示例充分 handel 了 AdaptDL 调度程序。 (\ ``main_adl.py`` 和 ``config_adl.yaml``\ )

这是将 AdaptDL 用作训练平台的模板配置规范。

.. code-block:: yaml

   authorName: default
   experimentName: minimal_adl

   trainingServicePlatform: adl
   nniManagerIp: 10.1.10.11
   logCollection: http

   tuner:
     builtinTunerName: GridSearch
   searchSpacePath: search_space.json

   trialConcurrency: 2
   maxTrialNum: 2

   trial:
     adaptive: false # optional.
     image: <image_tag>
     imagePullSecrets:  # optional
       - name: stagingsecret
     codeDir: .
     command: python main.py
     gpuNum: 1
     cpuNum: 1  # optional
     memorySize: 8Gi  # optional
     nfs: # optional
       server: 10.20.41.55
       path: /
       containerMountPath: /nfs
     checkpoint: # optional
       storageClass: dfs
       storageSize: 1Gi

下文中没有提及的 config 可以参考这篇文档：
`default specs defined in the NNI doc </Tutorial/ExperimentConfig.html#configuration-spec>`__。


* **trainingServicePlatform**\ : 选择 ``adl`` 以将 Kubernetes 集群与 AdaptDL 调度程序一起使用。
* **nniManagerIp**\ : *必填* ，为了 ``adl`` 训练平台能从群集中获取正确的信息和 metric 。
  具有启动 NNI 实验的 NNI 管理器（NNICTL）的计算机的IP地址。
* **logCollection**\ : *推荐* 设置 ``http``。 它将通过 http 将群集上的 trial log 收集到计算机。
* **tuner**\ : 支持 Tuun tuner 和所有的 NNI built-in tuners （仅限于 NNI PBT tuners 的 checkpoint 功能）。
* **trial**\ : 定义了 ``adl`` trial 的规格。

  * **namespace**\: （*可选*\ ） Kubernetes 命名空间启动 trial。 默认值是 ``default``。
  * **adaptive**\ : （*可选*\ ） 是否开启 AdaptDL trainer。 设置为 ``true``，这项工作是抢占性和适应性的。
  * **image**\ : trial 的 docker image。
  * **imagePullSecret**\ : （*可选*\ ） 如果使用私人注册表，
    需要提供密码才能成功提取 image。
  * **codeDir**\ : 容器的工作目录。 ``.`` 意味着默认的工作目录是 image 定义的。
  * **command**\ : 启动 trial 的 bash 命令。
  * **gpuNum**\ : trial 需要一系列 GPUs。 必须是非负整数。
  * **cpuNum**\ : （*可选*\ ） trial 需要一系列 CPUs。  必须是非负整数。
  * **memorySize**\ : （*可选*\ ） trial 需要的内存大小。 需要按照 Kubernetes 来。
    `默认设置 <https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory>`__。
  * **nfs**\ : （*可选*\ ） 安装外部存储。 使用 NFS 的详情请看下文。
  * **checkpoint** （*可选*\ ） 模型检查点的存储设置。

    * **storageClass**\ : 有关如何使用 ``storageClass`` 请参考 `Kubernetes storage 文档 <https://kubernetes.io/docs/concepts/storage/storage-classes/>`__ 。
    * **storageSize**\ : 此值应足够大以适合模型的检查点，否则可能导致 "disk quota exceeded" 错误。

NFS 存储
^^^^^^^^^^^

可能已经在上述配置规范中注意到，
*可选* 部分可用于配置 NFS 外部存储。 当不需要外部存储时，例如 docker image 足以容纳代码和数据时，它是可选的。

请注意，``adl`` 训练平台不能把 NFS 挂载到本地开发机器上，因此可以手动将 NFS 挂载到本地，管理文件系统，复制数据或代码等。
然后，使用适当的配置，``adl`` 训练平台可以针对每个 trial 将其安装到 kubernetes：


* **server**\ : NFS 服务地址，如 IP 地址或者 domain。
* **path**\ : NFS 服务导出路径，如 NFS 中可以安装到 trials 的绝对路径。
* **containerMountPath**\ : 在要安装上述 NFS **path** 的容器绝对路径中，
  以便于每条 trial 都可以连上 NFS。
  在每个 trial 的容器中，可以用这个路径去连接 NFS。

用例：


* 如果训练 trials 依赖于大型数据集，则可能需要先将其下载到NFS上，
  并安装它，以便可以在多个试用版之间共享。


* 容器的存储是临时性的，在试用期结束后，将删除 trial 容器。
  因此，如果要导出训练的模型，
  可以将NFS安装到试用版上，以保留并导出训练的模型。

简而言之，并没有限制 trial 如何读取或写入 NFS 存储，因此可以根据需要灵活使用它。

通过日志流监控
----------------------

遵循特定 trial 的日志流：

.. code-block:: bash

   nnictl log trial --trial_id=<trial_id>

.. code-block:: bash

   nnictl log trial <experiment_id> --trial_id=<trial_id>

请注意，在 trial 结束且其窗格已删除后，
无法通过该命令检索日志。
但是，仍然可以访问过去的试用记录
根据以下方法。

通过 TensorBoard 进行监控
----------------------------------------------

在 NNI 的背景下，一个实验有多条 trial。
为了在模型调整过程的各个 trial 之间轻松进行比较，
我们支持 TensorBoard 集成。 这里有一个实验
一个独立的 TensorBoard 日志目录，即 dashboard。

当被监控的实验处于 running 状态时你可以使用  TensorBoard。
换言之，不支持监视已经停止的实验。

在 trial 容器中，可以访问两个环境变量：


* ``ADAPTDL_TENSORBOARD_LOGDIR``\ : 当前实验  TensorBoard 日志目录，
* ``NNI_TRIAL_JOB_ID``\ : 当前 ``trial`` 的 job id。

建议将它们作为 trial 目录加入，
以 Python 举例：

.. code-block:: python

   import os
   tensorboard_logdir = os.path.join(
       os.getenv("ADAPTDL_TENSORBOARD_LOGDIR"),
       os.getenv("NNI_TRIAL_JOB_ID")
   )

如果实验停止，记录在此处的数据
（由 *以上envs* 定义，用于使用以下命令进行监视）
会丢掉。 要保留记录的数据，可以使用外部存储设备（例如 安装 NFS)
导出并在本地查看 TensorBoard。

通过上述设置，可以通过 TensorBoard 轻松监控实验。
 

.. code-block:: bash

   nnictl tensorboard start

如果有很多实验同时运行的话，可以使用

.. code-block:: bash

   nnictl tensorboard start <experiment_id>

将提供访问 tensorboard 的Web URL。

请注意，可以灵活地为 tensorboard 设置本地 ``--port`` 。
 
