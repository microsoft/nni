如何将自定义的 Tuner 安装为内置 Tuner
==================================================

参考下列步骤将自定义 Tuner： ``nni/examples/tuners/customized_tuner`` 安装为内置 Tuner。

准备安装源和安装包
-----------------------------------------------

有两种方法安装自定义的 Tuner：

方法 1: 从目录安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

步骤 1: 在 ``nni/examples/tuners/customized_tuner`` 目录下，运行：

``python setup.py develop``

此命令会将 ``nni/examples/tuners/customized_tuner`` 目录编译为 pip 安装源。

步骤 2: 运行命令

``nnictl package install ./``

方法 2: 从 whl 文件安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

步骤 1: 在 ``nni/examples/tuners/customized_tuner`` 目录下，运行：

``python setup.py bdist_wheel``

此命令会从 pip 安装源编译出 whl 文件。

步骤 2: 运行命令

``nnictl package install dist/demo_tuner-0.1-py3-none-any.whl``

检查安装的包
---------------------------

运行命令 ``nnictl package list``，可以看到已安装的 demotuner：

.. code-block:: bash

   +-----------------+------------+-----------+--------=-------------+------------------------------------------+
   |      Name       |    Type    | Installed |      Class Name      |               Module Name                |
   +-----------------+------------+-----------+----------------------+------------------------------------------+
   | demotuner       | tuners     | Yes       | DemoTuner            | demo_tuner                               |
   +-----------------+------------+-----------+----------------------+------------------------------------------+

在 Experiment 中使用安装的 Tuner
-------------------------------------

可以像使用其它内置 Tuner 一样，在 Experiment 配置文件中使用 demotuner：

.. code-block:: yaml

   tuner:
     builtinTunerName: demotuner
     classArgs:
       #choice: maximize, minimize
       optimize_mode: maximize
