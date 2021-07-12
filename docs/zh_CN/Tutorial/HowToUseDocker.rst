如何在 NNI 中使用 Docker
================================

概述
--------

`Docker <https://www.docker.com/>`__ 是一种工具, 可通过启动容器, 使用户能够更轻松地根据自己的操作系统部署和运行应用程序。 Docker 不是虚拟机，它不创建虚拟操作系统，但它允许不同的应用程序使用相同的操作系统内核，并通过容器隔离不同的应用程序。

用户可使用 Docker 来启动 NNI Experiment。 NNI 在 Docker Hub 上也提供了官方的 Docker 映像 `msranni/nni <https://hub.docker.com/r/msranni/nni>`__ 。

在本机使用docker
-----------------------------

第一步：Docker 的安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在开始使用 Docker 运行 NNI Experiment 前，首先需要在本机安装 Docker 运行程序。 `参考这里 <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`__。

第二步：启动 Docker 容器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果已经在本地机器上安装了 Docker 程序，可以启动 Docker 容器来运行 NNI 示例。 因为 NNI 会在 Docker 容器里启动 Web 界面进程，并监听端口，因此需要指定一个在主机和 Docker 容器映射的端口，可在容器外访问 Docker 容器里的进程。 通过访问主机的 IP 和端口，就可以访问容器里的 Web 网页进程了。

例如，通过如下命令来启动 Docker 容器：

.. code-block:: bash

   docker run -i -t -p [hostPort]:[containerPort] [image]

``-i:`` 使用交互模式启动 Docker。

``-t:`` 为 Docker 分配一个输入终端。

``-p:`` 端口映射，映射主机端口和容器端口。

可以参考 `这里 <https://docs.docker.com/v17.09/edge/engine/reference/run/>`__，获取更多的命令参考。

注意：

.. code-block:: bash

      NNI 目前仅支持本机模式下的 Ubuntu 和 macOS 系统，请使用正确的 Docker 映像类型。 如果想要在 Docker 容器里面使用 GPU，请使用 nvidia-docker。

第三步：在 Docker 容器里运行 NNI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果直接使用 NNI 的官方镜像 ``msranni/nni`` 来启动 Experiment，可以直接使用 ``nnictl`` 命令。 NNI 官方镜像有最基础的 Python 环境和深度学习框架。

如果使用自己的 Docker 镜像，需要首先 `安装 NNI <InstallationLinux.rst>`__。

如果要使用 NNI 的官方示例，可以通过以下 git 命令来克隆 NNI：

.. code-block:: bash

   git clone https://github.com/Microsoft/nni.git

然后可以进入 ``nni/examples/trials`` 文件夹来启动 Experiment。

准备好 NNI 的环境后，可使用 ``nnictl`` 命令开始新的 Experiment。 `入门教程 <QuickStart.rst>`__。

在远程平台上运行 Docker
---------------------------------

NNI 支持在 `远程平台 <../TrainingService/RemoteMachineMode.rst>`__ 上启动 Experiment，并在远程机器里运行 Trial。 因为 Docker 可以运行独立的 Ubuntu 系统和 SSH 服务，因此 Docker 容器可以作为远程平台来运行 NNI。

第一步：设置 Docker 环境
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

你首先应该在远程机器上安装docker工具，`参考这里 <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`__。

为保证 Docker 容器可以被 NNI Experiment 连接上，要在自己的 Docker 容器里安装 SSH 服务，或使用已经配置好 SSH 的映像。 如果要在 Docker 容器里使用 SSH 服务，需要配置 SSH 密码登录或者私钥登录，`参考这里 <https://docs.docker.com/engine/examples/running_ssh_service/>`__。

注意：

.. code-block:: text

   NNI 的官方镜像 msranni/nni 暂不支持 SSH 服务，应构建自己的带有 SSH 服务的映像，或者使用其他的带有 SSH 服务的镜像。

第二步：在远程机器上启动 Dokcer 容器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSH 服务需要端口，要把 Docker 的 SSH 服务端口暴露给 NNI 作为连接端口。 例如，如果设置容器的端口 ``A`` 作为 SSH 端口，应把端口 ``A`` 映射到主机的端口 ``B``，NNI 会连接端口 ``B`` 作为 SSH 服务端口，主机会把连接到端口 ``B`` 的连接映射到端口 ``A``，NNI 就可以连接到容器中了。

例如，通过如下命令来启动 Docker 容器：

.. code-block:: bash

   docker run -dit -p [hostPort]:[containerPort] [image]

``containerPort`` 是在 Docker 容器中指定的端口，``hostPort`` 是主机的端口。 可设置 NNI 配置，连接到``hostPort``，这个连接会被转发到 Docker 容器。
可以参考 `这里 <https://docs.docker.com/v17.09/edge/engine/reference/run/>`__，获取更多的命令参考。

注意：

.. code-block:: bash

   如果使用自己构建的 Docker 映像，确保有基础的 Python 运行时和 NNI SDK 环境。 如果想要在 Docker 容器里面使用 GPU，请使用 nvidia-docker。

第三步：运行 NNI Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以在配置文件中，设置训练平台为远程平台，然后设置 ``machineList`` 配置，`参考这里 <../TrainingService/RemoteMachineMode.rst>`__。 注意应该设置正确的 ``port``\ , ``username``\ , 以及 ``passWd`` 或 ``sshKeyPath`` 。

``port:`` 主机的端口，映射到 Docker 的 SSH 端口。

``username:`` Docker 容器的用户名。

``passWd:`` Docker 容器的密码。

``sshKeyPath:`` Docker 容器私钥的存储路径。

设置完配置文件，你就可以启动实验了，`参考这里 <QuickStart.rst>`__。
