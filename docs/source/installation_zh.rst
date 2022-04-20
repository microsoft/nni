.. b4703fc8c8e8dc1babdb38ba9ebcd4a6

安装 NNI
========

NNI 依赖于 Python 3.7 或以上版本。

您可以通过以下三种方式之一安装 NNI：

* :ref:`通过 pip 安装<zh-installation-pip>`
* :ref:`从源代码编译安装<zh-installation-source>`
* :ref:`使用 Docker 容器<zh-installation-docker>`

.. _zh-installation-pip:

pip 安装
--------

NNI 为 x86-64 平台提供预编译的安装包，您可以使用 pip 进行安装：

.. code-block:: text

    pip install nni

您也可以升级已安装的旧版本 NNI：

.. code-block:: text

    pip install --latest nni

安装完成后，请运行以下命令进行检查：

.. code-block:: text

    nnictl --version

如果您使用的是 Linux 系统并且没有使用 Conda，您可能会遇到 ``bash: nnictl: command not found`` 错误，
此时您需要将 pip 安装的可执行文件添加到 ``PATH`` 环境变量：

.. code-block:: bash

    echo 'export PATH=${PATH}:${HOME}/.local/bin' >> ~/.bashrc
    source ~/.bashrc

.. _zh-installation-source:

编译安装
--------

NNI 项目使用 `GitHub <https://github.com/microsoft/nni>`__ 托管源代码。

NNI 对 ARM64 平台（包括苹果 M1）提供实验性支持，如果您希望在此类平台上使用 NNI，请从源代码编译安装。

编译步骤请参见英文文档： :doc:`/notes/build_from_source`

.. _zh-installation-docker:

Docker 镜像
-----------

NNI 在 `Docker Hub <https://hub.docker.com/r/msranni/nni>`__ 上提供了官方镜像。

.. code-block:: text

    docker pull msranni/nni

安装额外依赖
------------

有一些算法依赖于额外的 pip 包，在使用前需要先指定 ``nni[算法名]`` 安装依赖。以 DNGO 算法为例，使用前请运行以下命令：

.. code-block:: text

    pip install nni[DNGO]

如果您已经通过任一种方式安装了 NNI，以上命令不会重新安装或改变 NNI 版本，只会安装 DNGO 算法的额外依赖。

您也可以一次性安装所有可选依赖：

.. code-block:: text

    pip install nni[all]

**注意**：SMAC 算法依赖于 swig3，在 Ubuntu 系统中需要手动进行降级：

.. code-block:: bash

    sudo apt install swig3.0
    sudo rm /usr/bin/swig
    sudo ln -s swig3.0 /usr/bin/swig
