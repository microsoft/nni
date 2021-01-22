.. role:: raw-html(raw)
   :format: html


nnictl
======

介绍
------------

**nnictl** 是一个命令行工具，用来控制 NNI Experiment，如启动、停止、继续 Experiment，启动、停止 NNIBoard 等等。

命令
--------

nnictl 支持的命令：


* `nnictl create <#create>`__
* `nnictl resume <#resume>`__
* `nnictl view <#view>`__
* `nnictl stop <#stop>`__
* `nnictl update <#update>`__
* `nnictl trial <#trial>`__
* `nnictl top <#top>`__
* `nnictl experiment <#experiment>`__
* `nnictl platform <#platform>`__
* `nnictl config <#config>`__
* `nnictl log <#log>`__
* `nnictl webui <#webui>`__
* `nnictl tensorboard <#tensorboard>`__
* `nnictl package <#package>`__
* `nnictl ss_gen <#ss_gen>`__
* `nnictl --version <#version>`__

管理 Experiment
^^^^^^^^^^^^^^^^^^^^

:raw-html:`<a name="create"></a>`

nnictl create
^^^^^^^^^^^^^


* 
  说明

  此命令使用参数中的配置文件，来创建新的实验。

  此命令成功完成后，上下文会被设置为此 Experiment。这意味着如果不显式改变上下文（暂不支持），输入的以下命令，都作用于此 Experiment。

* 
  用法

  .. code-block:: bash

     nnictl create [OPTIONS]

* 
  选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - --config, -c
     - True
     - 
     - Experiment 的 YAML 配置文件
   * - --port, -p
     - False
     - 
     - RESTful 服务的端口
   * - --debug, -d
     - False
     - 
     - 设置为调试模式
   * - --foreground, -f
     - False
     - 
     - 设为前台运行模式，将日志输出到终端



* 
  示例

  ..

     在默认端口 8080 上创建一个新的 Experiment


  .. code-block:: bash

     nnictl create --config nni/examples/trials/mnist-tfv1/config.yml

  ..

     在指定的端口 8088 上创建新的 Experiment


  .. code-block:: bash

     nnictl create --config nni/examples/trials/mnist-tfv1/config.yml --port 8088

  ..

     在指定的端口 8088 上创建新的 Experiment，并启用调试模式


  .. code-block:: bash

     nnictl create --config nni/examples/trials/mnist-tfv1/config.yml --port 8088 --debug

注意：

.. code-block:: text

   调试模式会禁用 Trialkeeper 中的版本校验功能。

:raw-html:`<a name="resume"></a>`

nnictl resume
^^^^^^^^^^^^^


* 
  说明

  使用此命令恢复已停止的 Experiment。

* 
  用法

  .. code-block:: bash

     nnictl resume [OPTIONS]

* 
  选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - True
     - 
     - 要恢复的 Experiment 标识
   * - --port, -p
     - False
     - 
     - 要恢复的 Experiment 使用的 RESTful 服务端口
   * - --debug, -d
     - False
     - 
     - 设置为调试模式
   * - --foreground, -f
     - False
     - 
     - 设为前台运行模式，将日志输出到终端



* 
  示例

  ..

     在指定的端口 8088 上恢复 Experiment


  .. code-block:: bash

     nnictl resume [experiment_id] --port 8088

:raw-html:`<a name="view"></a>`

nnictl view
^^^^^^^^^^^


* 
  说明

  使用此命令查看已停止的 Experiment。

* 
  用法

  .. code-block:: bash

     nnictl view [OPTIONS]

* 
  选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - True
     - 
     - 要查看的 Experiment 标识
   * - --port, -p
     - False
     - 
     - 要查看的 Experiment 使用的 RESTful 服务端口



* 
  示例

  ..

     在指定的端口 8088 上查看 Experiment


  .. code-block:: bash

     nnictl view [experiment_id] --port 8088

:raw-html:`<a name="stop"></a>`

nnictl stop
^^^^^^^^^^^


* 
  说明

  使用此命令来停止正在运行的单个或多个 Experiment。

* 
  用法

  .. code-block:: bash

     nnictl stop [Options]

* 
  选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 要停止的 Experiment 标识
   * - --port, -p
     - False
     - 
     - 要停止的 Experiment 使用的 RESTful 服务端口
   * - --all, -a
     - False
     - 
     - 停止所有 Experiment



* 
  详细信息及样例


  #. 
     如果没有指定 id，并且当前有运行的 Experiment，则会停止该 Experiment，否则会输出错误信息。

     .. code-block:: bash

         nnictl stop

  #. 
     如果指定了 id，并且此 id 匹配正在运行的 Experiment，nnictl 会停止相应的 Experiment，否则会输出错误信息。

     .. code-block:: bash

         nnictl stop [experiment_id]

  #. 
     如果指定了端口，并且此端口有正在运行的 Experiment，则会停止此 Experiment。

     .. code-block:: bash

         nnictl stop --port 8080

  #. 
     可使用 'nnictl stop --all' 来停止所有的 Experiment。

     .. code-block:: bash

         nnictl stop --all

  #. 
     如果 id 以 * 结尾，nnictl 会停止所有匹配此通配符的 Experiment。

  #. 如果 id 不存在，但匹配了某个Experiment 的 id 前缀，nnictl 会停止匹配的Experiment 。
  #. 如果 id 不存在，但匹配多个 Experiment id 的前缀，nnictl 会输出这些 id 的信息。

:raw-html:`<a name="update"></a>`

nnictl update
^^^^^^^^^^^^^


* 
  **nnictl update searchspace**


  * 
    说明

    可以用此命令来更新 Experiment 的搜索空间。

  * 
    用法

    .. code-block:: bash

       nnictl update searchspace [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --filename, -f
     - True
     - 
     - 新的搜索空间文件名



* 
  示例

  ``使用 'examples/trials/mnist-tfv1/search_space.json' 来更新 Experiment 的搜索空间``

  .. code-block:: bash

     nnictl update searchspace [experiment_id] --filename examples/trials/mnist-tfv1/search_space.json


* 
  **nnictl update concurrency**


  * 
    说明

     可以用此命令来更新 Experiment 的并发设置。

  * 
    用法

    .. code-block:: bash

       nnictl update concurrency [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --value, -v
     - True
     - 
     - 允许同时运行的 Trial 的数量



* 
  示例

  ..

     更新 Experiment 的并发数量


  .. code-block:: bash

     nnictl update concurrency [experiment_id] --value [concurrency_number]


* 
  **nnictl update duration**


  * 
    说明

    可以用此命令来更新实验的运行时间。

  * 
    用法

    .. code-block:: bash

       nnictl update duration [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --value, -v
     - True
     - 
     - 形如 '1m' （一分钟）或 '2h' （两小时）的字符串。 后缀可以为 's'（秒）, 'm'（分钟）, 'h'（小时）或 'd'（天）。



* 
  示例

  ..

     修改 Experiment 的执行时间


  .. code-block:: bash

     nnictl update duration [experiment_id] --value [duration]


* 
  **nnictl update trialnum**


  * 
    说明

    可以用此命令来更新实验的最大尝试数量。

  * 
    用法

    .. code-block:: bash

       nnictl update trialnum [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --value, -v
     - True
     - 
     - 需要设置的 maxtrialnum 的数量



* 
  示例

  ..

     更新 Experiment 的 Trial 数量


  .. code-block:: bash

     nnictl update trialnum [experiment_id] --value [trial_num]

:raw-html:`<a name="trial"></a>`

nnictl trial
^^^^^^^^^^^^


* 
  **nnictl trial ls**


  * 
    说明

    使用此命令来查看 Trial 的信息。 注意如果 ``head`` 或 ``tail`` 被设置, 则只有完成的 Trial 会被展示。

  * 
    用法

    .. code-block:: bash

       nnictl trial ls
       nnictl trial ls --head 10
       nnictl trial ls --tail 10

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --head
     - False
     - 
     - 依据最高默认指标列出的项数。
   * - --tail
     - False
     - 
     - 依据最低默认指标列出的项数。



* 
  **nnictl trial kill**


  * 
    说明

    此命令用于终止 Trial。

  * 
    用法

    .. code-block:: bash

       nnictl trial kill [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - Trial 的 Experiment ID
   * - --trial_id, -T
     - True
     - 
     - 需要终止的 Trial 的 ID。



* 
  示例

  ..

     结束 Trial 任务


  .. code-block:: bash

     nnictl trial kill [experiment_id] --trial_id [trial_id]

:raw-html:`<a name="top"></a>`

nnictl top
^^^^^^^^^^


* 
  说明

  查看正在运行的 Experiment。

* 
  用法

  .. code-block:: bash

     nnictl top

* 
  选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --time, -t
     - False
     - 
     - 刷新 Experiment 状态的时间间隔，单位为秒，默认值为 3 秒。


:raw-html:`<a name="experiment"></a>`

管理 Experiment 信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  **nnictl experiment show**


  * 
    说明

    显示 Experiment 的信息。

  * 
    用法

    .. code-block:: bash

       nnictl experiment show

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id



* 
  **nnictl experiment status**


  * 
    说明

    显示 Experiment 的状态。

  * 
    用法

    .. code-block:: bash

       nnictl experiment status

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id



* 
  **nnictl experiment list**


  * 
    说明

    显示正在运行的 Experiment 的信息

  * 
    用法

    .. code-block:: bash

       nnictl experiment list [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - --all
     - False
     - 
     - 列出所有 Experiment



* 
  **nnictl experiment delete**


  * 
    说明

    删除一个或所有 Experiment，包括日志、结果、环境信息和缓存。 用于删除无用的 Experiment 结果，或节省磁盘空间。

  * 
    用法

    .. code-block:: bash

       nnictl experiment delete [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - Experiment ID
   * - --all
     - False
     - 
     - 删除所有 Experiment



* 
  **nnictl experiment export**


  * 
    说明

    使用此命令，可将 Trial 的 reward 和超参导出为 csv 文件。

  * 
    用法

    .. code-block:: bash

       nnictl experiment export [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - Experiment ID
   * - --filename, -f
     - True
     - 
     - 文件的输出路径
   * - --type
     - True
     - 
     - 输出文件类型，仅支持 "csv" 和 "json"
   * - --intermediate, -i
     - False
     - 
     - 是否保存中间结果



* 
  示例

  ..

     将 Experiment 中所有 Trial 数据导出为 JSON 格式


  .. code-block:: bash

     nnictl experiment export [experiment_id] --filename [file_path] --type json --intermediate


* 
  **nnictl experiment import**


  * 
    说明

    可使用此命令将以前的 Trial 超参和结果导入到 Tuner 中。 数据会传入调参算法中（即 Tuner 或 Advisor）。

  * 
    用法

    .. code-block:: bash

       nnictl experiment import [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要将数据导入的 Experiment 的 ID
   * - --filename, -f
     - True
     - 
     - 需要导入的 JSON 格式的数据文件



* 
  详细说明

  NNI 支持导入用户的数据，确保数据格式正确。 样例如下：

  .. code-block:: json

     [
       {"parameter": {"x": 0.5, "y": 0.9}, "value": 0.03},
       {"parameter": {"x": 0.4, "y": 0.8}, "value": 0.05},
       {"parameter": {"x": 0.3, "y": 0.7}, "value": 0.04}
     ]

  最顶层列表的每个元素都是一个样例。 对于内置的 Tuner 和 Advisor，每个样本至少需要两个主键：``parameter`` 和 ``value``。 ``parameter`` 必须与 Experiment 的搜索空间相匹配，``parameter`` 中的所有的主键（或超参）都必须与搜索空间中的主键相匹配。 否则， Tuner 或 Advisor 可能会有无法预期的行为。 ``Value`` 应当遵循与 ``nni.report_final_result`` 的输入值一样的规则，即要么时一个数字，或者是包含 ``default`` 主键的 dict。 对于自定义的 Tuner 或 Advisor，根据实现的不同，此文件可以是任意的 JSON 内容（例如，``import_data``）。

  也可以用 `nnictl experiment export <#export>`__ 命令导出 Experiment 已经运行过的 Trial 超参和结果。

  当前，以下 Tuner 和 Advisor 支持导入数据：

  .. code-block:: yaml

     builtinTunerName: TPE, Anneal, GridSearch, MetisTuner
     builtinAdvisorName: BOHB

  *如果要将数据导入到 BOHB Advisor，建议像 NNI 一样，增加 "TRIAL_BUDGET" 参数，否则，BOHB 会使用 max_budget 作为 "TRIAL_BUDGET"。* 示例如下：

  .. code-block:: json

     [
       {"parameter": {"x": 0.5, "y": 0.9, "TRIAL_BUDGET": 27}, "value": 0.03}
     ]

* 
  示例

  ..

     将数据导入运行中的 Experiment


  .. code-block:: bash

     nnictl experiment import [experiment_id] -f experiment_data.json


* 
  **nnictl experiment save**


  * 
    说明

    保存 NNI Experiment 的元数据及代码数据

  * 
    用法

    .. code-block:: bash

       nnictl experiment save [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - True
     - 
     - 要保存的 Experiment 标识
   * - --path, -p
     - False
     - 
     - 保存 NNI Experiment 数据的路径，默认为当前工作目录
   * - --saveCodeDir, -s
     - False
     - 
     - 是否保存 Experiment 的代码目录的数据，默认为 False



* 
  示例

  ..

     保存 Experiment


  .. code-block:: bash

     nnictl experiment save [experiment_id] --saveCodeDir


* 
  **nnictl experiment load**


  * 
    说明

    加载 NNI Experiment

  * 
    用法

    .. code-block:: bash

       nnictl experiment load [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - --path, -p
     - True
     - 
     - NNI 包的文件路径
   * - --codeDir, -c
     - True
     - 
     - 要加载的实验的代码目录，加载的 NNI 包中的代码也会放到此目录下。
   * - --logDir, -l
     - False
     - 
     - 存放加载的实验的日志的目录。
   * - --searchSpacePath, -s
     - True
     - 
     - 存放加载的实验的搜索空间文件的路径（路径包含文件名）。 默认是 $codeDir/search_space.json。



* 
  示例

  ..

     加载 Experiment


  .. code-block:: bash

     nnictl experiment load --path [path] --codeDir [codeDir]

:raw-html:`<a name="platform"></a>`

管理平台信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  **nnictl platform clean**


  * 
    说明

    用于清理目标平台上的磁盘空间。 所提供的 YAML 文件包括了目标平台的信息，与 NNI 配置文件的格式相同。

  * 
    注意

    如果目标平台正在被别人使用，可能会造成他人的意外错误。

  * 
    用法

    .. code-block:: bash

       nnictl platform clean [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - --config
     - True
     - 
     - 创建 Experiment 时的 YAML 配置文件路径。


:raw-html:`<a name="config"></a>`

nnictl config show
^^^^^^^^^^^^^^^^^^


* 
  说明

  显示当前上下文信息。

* 
  用法

  .. code-block:: bash

     nnictl config show

:raw-html:`<a name="log"></a>`

管理日志
^^^^^^^^^^


* 
  **nnictl log stdout**


  * 
    说明

    显示 stdout 日志内容。

  * 
    用法

    .. code-block:: bash

       nnictl log stdout [options]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --head, -h
     - False
     - 
     - 显示 stdout 开始的若干行
   * - --tail, -t
     - False
     - 
     - 显示 stdout 结尾的若干行
   * - --path, -p
     - False
     - 
     - 显示 stdout 文件的路径



* 
  示例

  ..

     显示 stdout 结尾的若干行


  .. code-block:: bash

     nnictl log stdout [experiment_id] --tail [lines_number]


* 
  **nnictl log stderr**


  * 
    说明

    显示 stderr 日志内容。

  * 
    用法

    .. code-block:: bash

       nnictl log stderr [options]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --head, -h
     - False
     - 
     - 显示 stderr 开始的若干行
   * - --tail, -t
     - False
     - 
     - 显示 stderr 结尾的若干行
   * - --path, -p
     - False
     - 
     - 显示 stderr 文件的路径



* 
  **nnictl log trial**


  * 
    说明

    显示 Trial 日志的路径。

  * 
    用法

    .. code-block:: bash

       nnictl log trial [options]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - Trial 的 Experiment ID
   * - --trial_id, -T
     - False
     - 
     - 所需要找日志路径的 Trial 的 ID，当 id 不为空时，此值也为必需。


:raw-html:`<a name="webui"></a>`

Manage webui
^^^^^^^^^^^^


* 
  **nnictl webui url**


  * 
    说明

    显示 Experiment 的 Web 界面链接

  * 
    用法

    .. code-block:: bash

       nnictl webui url [options]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - Experiment ID


:raw-html:`<a name="tensorboard"></a>`

管理 tensorboard
^^^^^^^^^^^^^^^^^^


* 
  **nnictl tensorboard start**


  * 
    说明

    启动 tensorboard 进程。

  * 
    用法

    .. code-block:: bash

       nnictl tensorboard start

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id
   * - --trial_id, -T
     - False
     - 
     - Trial 的 id
   * - --port
     - False
     - 6006
     - tensorboard 进程的端口



* 
  详细说明


  #. NNICTL 当前仅支持本机和远程平台的 tensorboard，其它平台暂不支持。
  #. 如果要使用 tensorboard，需要将 tensorboard 日志输出到环境变量 [NNI_OUTPUT_DIR] 路径下。
  #. 在 local 模式中，nnictl 会直接设置 --logdir=[NNI_OUTPUT_DIR] 并启动 tensorboard 进程。
  #. 在 remote 模式中，nnictl 会创建一个 ssh 客户端来将日志数据从远程计算机复制到本机临时目录中，然后在本机开始 tensorboard 进程。 需要注意的是，nnictl 只在使用此命令时复制日志数据，如果要查看最新的 tensorboard 结果，需要再次执行 nnictl tensorboard 命令。
  #. 如果只有一个 Trial 任务，不需要设置 Trial ID。 如果有多个运行的 Trial 作业，需要设置 Trial ID，或使用 [nnictl tensorboard start --trial_id all] 来将 --logdir 映射到所有 Trial 的路径。


* 
  **nnictl tensorboard stop**


  * 
    说明

    停止所有 tensorboard 进程。

  * 
    用法

    .. code-block:: bash

       nnictl tensorboard stop

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - id
     - False
     - 
     - 需要设置的 Experiment 的 id


:raw-html:`<a name="package"></a>`

管理安装包
^^^^^^^^^^^^^^


* 
  **nnictl package install**


  * 
    说明

    安装自定义的 Tuner，Assessor，Advisor（定制或 NNI 提供的算法）。

  * 
    用法

    .. code-block:: bash

       nnictl package install --name <package name>

    可通过 ``nnictl package list`` 命令查看可用的 ``<包名称>``。

    或者

    .. code-block:: bash

       nnictl package install <安装源>

    参考 `安装自定义算法 <InstallCustomizedAlgos.rst>`__ 来准备安装源。

  * 
    示例

    ..

       安装 SMAC Tuner


    .. code-block:: bash

       nnictl package install --name SMAC

    ..

       安装自定义 Tuner


    .. code-block:: bash

       nnictl package install nni/examples/tuners/customized_tuner/dist/demo_tuner-0.1-py3-none-any.whl


* 
  **nnictl package show**


  * 
    说明

    显示包的详情。

  * 
    用法

    .. code-block:: bash

       nnictl package show <包名称>

  * 
    示例

    .. code-block:: bash

       nnictl package show SMAC

* 
  **nnictl package list**


  * 
    说明

    列出已安装的包 / 所有包。

  * 
    用法

    .. code-block:: bash

       nnictl package list [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - --all
     - False
     - 
     - 列出所有包



* 
  示例

  ..

     列出已安装的包


  .. code-block:: bash

     nnictl package list

  ..

     列出所有包


  .. code-block:: bash

     nnictl package list --all


* 
  **nnictl package uninstall**


  * 
    说明

    卸载包。

  * 
    用法

    .. code-block:: bash

       nnictl package uninstall <包名称>

  * 
    示例
    卸载 SMAC 包

    .. code-block:: bash

       nnictl package uninstall SMAC

:raw-html:`<a name="ss_gen"></a>`

生成搜索空间
^^^^^^^^^^^^^^^^^^^^^


* 
  **nnictl ss_gen**


  * 
    说明

    从使用 NNI NAS API 的用户代码生成搜索空间。

  * 
    用法

    .. code-block:: bash

       nnictl ss_gen [OPTIONS]

  * 
    选项

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 参数及缩写
     - 是否必需
     - 默认值
     - 说明
   * - --trial_command
     - True
     - 
     - Trial 代码的命令
   * - --trial_dir
     - False
     - ./
     - Trial 代码目录
   * - --file
     - False
     - nni_auto_gen_search_space.json
     - 用来存储生成的搜索空间



* 
  示例

  ..

     生成搜索空间


  .. code-block:: bash

     nnictl ss_gen --trial_command="python3 mnist.py" --trial_dir=./ --file=ss.json

:raw-html:`<a name="version"></a>`

NNI 版本校验
^^^^^^^^^^^^^^^^^


* 
  **nnictl --version**


  * 
    说明

    显示当前安装的 NNI 的版本。

  * 
    用法

    .. code-block:: bash

       nnictl --version
