如何使用共享存储
=============================

如果您想在使用 NNI 期间使用自己的存储，共享存储可以满足您的需求。
与使用训练平台本机存储不同，共享存储可以为您带来更多便利。
Experiment 生成的所有信息都将存储在共享存储的 ``/nni`` 文件夹下。
Trial 产生的所有输出将位于共享存储中的 ``/nni/{EXPERIMENT_ID}/trials/{TRIAL_ID}/nnioutput`` 文件夹下。
这就避免了在不同地方寻找实验相关信息的麻烦。
Trial 工作目录是 ``/nni/{EXPERIMENT_ID}/trials/{TRIAL_ID}``，因此如果您在共享存储中上载数据，您可以像在 Trial 代码中打开本地文件一样打开它，而不必下载它。
未来我们将开发更多基于共享存储的实用功能。

.. note::
    共享存储目前处于实验阶段。 我们建议在 Ubuntu/CentOS/RHEL 下使用 AzureBlob，在 Ubuntu/CentOS/RHEL/Fedora/Debian 下使用 NFS 进行远程访问。
    确保您的本地机器可以挂载 NFS 或 fuse AzureBlob，并在远程运行时具有 sudo 权限。 我们目前只支持使用重用模式的训练平台下的共享存储。

示例
-------
如果要使用 AzureBlob，请在配置中添加以下内容。完整的配置文件请参阅 :githublink:`mnist-sharedstorage/config_azureblob.yml <examples/trials/mnist-sharedstorage/config_azureblob.yml>`。

.. code-block:: yaml

    sharedStorage:
        storageType: AzureBlob
        localMountPoint: ${your/local/mount/point}
        remoteMountPoint: ${your/remote/mount/point}
        storageAccountName: ${replace_to_your_storageAccountName}
        storageAccountKey: ${replace_to_your_storageAccountKey}
        # 如果未设置 storageAccountKey，则首先需要在 Azure CLI 中使用 `az login` 并设置 resourceGroupName。
        # resourceGroupName: ${replace_to_your_resourceGroupName}
        containerName: ${replace_to_your_containerName}
        # usermount 表示已将此存储挂载在 localMountPoint 上
        # nnimount 表示 NNI 将尝试将此存储挂载在 localMountPoint 上
        # nomount 表示存储不会挂载在本地机器上，将在未来支持部分存储。 
        localMounted: nnimount

如果要使用 NFS，请在配置中添加以下内容。完整的配置文件请参阅 :githublink:`mnist-sharedstorage/config_nfs.yml <examples/trials/mnist-sharedstorage/config_nfs.yml>`。

.. code-block:: yaml

    sharedStorage:
        storageType: NFS
        localMountPoint: ${your/local/mount/point}
        remoteMountPoint: ${your/remote/mount/point}
        nfsServer: ${nfs-server-ip}
        exportedDirectory: ${nfs/exported/directory}
        # usermount 表示已将此存储挂载在 localMountPoint 上
        # nnimount 表示 NNI 将尝试将此存储挂载在 localMountPoint 上
        # nomount 表示存储不会挂载在本地机器上，将在未来支持部分存储。 
        localMounted: nnimount
