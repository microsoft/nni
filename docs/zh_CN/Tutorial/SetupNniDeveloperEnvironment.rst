设置 NNI 开发环境
=================================

NNI 开发环境支持安装 Python 3 64 位的 Ubuntu 1604 （及以上）和 Windows 10。

安装
------------

1. 克隆源代码
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Microsoft/nni.git

注意，如果要贡献代码，需要 Fork 自己的 NNI 代码库并克隆。

2. 从源代码安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3 -m pip install --upgrade pip setuptools
   python3 setup.py develop

这是在 `开发模式 <https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html>`__ 下安装NNI，
所以你不需要在编辑之后重新安装。

3. 检查环境是否正确
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

尝试启动实验来检查环境。
例如，运行命令

.. code-block:: bash

   nnictl create --config examples/trials/mnist-pytorch/config.yml

并打开网页界面查看

4. 重新加载改动
^^^^^^^^^^^^^^^^^

Python
^^^^^^

无需操作，代码已连接到包的安装位置。

TypeScript (Linux 和 macOS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* 如果改动了 ``tc/nni_manager``，在此目录下运行 ``yarn watch`` 可持续编译改动。 它将持续的监视并编译代码。 可能需要重新启动 ``nnictl`` 来重新加载 NNI 管理器。
* 如果改动了 ``tx/webui`` ，运行 ``yarn dev``， 该命令将同时运行一个 模拟 API 服务器和一个 webpack 开发服务器。 使用环境变量 ``EXPERIMENT`` (例如 ``mnist-tfv1-running``\ ) 来指定要用到的模拟数据。 内置的模拟实验列在 ``src/webui/mock``。 完整示例：``EXPERIMENT=mnist-tfv1-running yarn dev``。
* 如果改动了 ``ts/nasui``，在相应目录下运行 ``yarn start``。 Web 界面会在代码修改后自动刷新。 还有一个在开发时有用的模拟 API 服务器， 可以通过 ``node server.js`` 来启动。

TypeScript (Windows)
^^^^^^^^^^^^^^^^^^^^

目前，您必须在编辑后使用 `python3 setup.py build_ts` 重建 TypeScript 模块。

5. 提交拉取请求
^^^^^^^^^^^^^^^^^^^^^^

所有改动都需要从自己 Fork 的代码库合并到 master 分支上。 拉取请求的描述必须有意义且有用。

我们会尽快审查更改。 审查通过后，我们会将代码合并到主分支。

有关更多贡献指南和编码风格，查看 `贡献文档 <Contributing.rst>`__。
