# nnictl

## 介绍

**nnictl** 是一个命令行工具，用来控制 NNI Experiment，如启动、停止、继续 Experiment，启动、停止 NNIBoard 等等。

## 命令

nnictl 支持的命令：

* [nnictl create](#create)
* [nnictl resume](#resume)
* [nnictl stop](#stop)
* [nnictl update](#update)
* [nnictl trial](#trial)
* [nnictl top](#top)
* [nnictl experiment](#experiment)
* [nnictl config](#config)
* [nnictl log](#log)
* [nnictl webui](#webui)
* [nnictl tensorboard](#tensorboard)
* [nnictl package](#package)
* [nnictl --version](#version)

### 管理 Experiment

<a name="create"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl create`

* 说明
  
  此命令使用参数中的配置文件，来创建新的 Experiment。
  
  此命令成功完成后，上下文会被设置为此 Experiment。这意味着如果不显式改变上下文（暂不支持），输入的以下命令，都作用于此 Experiment。

* 用法
  
  ```bash
  nnictl create [OPTIONS]
  ```

* 选项
  
  | 参数及缩写        | 是否必需  | 默认值 | 说明                     |
  | ------------ | ----- | --- | ---------------------- |
  | --config, -c | True  |     | Experiment 的 YAML 配置文件 |
  | --port, -p   | False |     | RESTful 服务的端口          |
  | --debug, -d  | False |     | 设置为调试模式                |

* 样例
  
  > 在默认端口 8080 上创建一个新的 Experiment
  
  ```bash
  nnictl create --config nni/examples/trials/mnist/config.yml
  ```
  
  > 在指定的端口 8088 上创建新的 Experiment
  
  ```bash
  nnictl create --config nni/examples/trials/mnist/config.yml --port 8088
  ```
  
  > 在指定的端口 8088 上创建新的 Experiment，并启用调试模式
  
  ```bash
  nnictl create --config nni/examples/trials/mnist/config.yml --port 8088 --debug
  ```

注意：

```text
调试模式会禁用 Trialkeeper 中的版本校验功能。
```

<a name="resume"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl resume`

* 说明
  
  使用此命令恢复已停止的 Experiment。

* 用法
  
  ```bash
  nnictl resume [OPTIONS]
  ```

* 选项
  
  | 参数及缩写       | 是否必需  | 默认值 | 说明                               |
  | ----------- | ----- | --- | -------------------------------- |
  | id          | True  |     | 要恢复的 Experiment 标识               |
  | --port, -p  | False |     | 要恢复的 Experiment 使用的 RESTful 服务端口 |
  | --debug, -d | False |     | 设置为调试模式                          |

* 样例
  
  > 在指定的端口 8088 上恢复 Experiment
  
  ```bash
  nnictl resume [experiment_id] --port 8088
  ```

<a name="stop"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl stop`

* 说明
  
  使用此命令来停止正在运行的单个或多个 Experiment。

* 用法
  
  ```bash
  nnictl stop [id]
  ```

* 详细信息及样例
  
  1. 如果没有指定 id，并且当前有运行的 Experiment，则会停止该 Experiment，否则会输出错误信息。
    
        ```bash
        nnictl stop
        ```
        
  
  2. 如果指定了 id，并且此 id 匹配正在运行的 Experiment，nnictl 会停止相应的 Experiment，否则会输出错误信息。
    
        ```bash
        nnictl stop [experiment_id]
        ```
        
  
  3. 可使用 'nnictl stop all' 来停止所有的 Experiment。
    
        ```bash
        nnictl stop all
        ```
        
  
  4. 如果 id 以 * 结尾，nnictl 会停止所有匹配此通配符的 Experiment。
  
  5. 如果 id 不存在，但匹配了某个Experiment 的 id 前缀，nnictl 会停止匹配的Experiment 。
  6. 如果 id 不存在，但匹配了多个 Experiment id 的前缀，nnictl 会输出这些 id 的信息。

<a name="update"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl update`

* **nnictl update searchspace**
  
  * 说明
    
    可以用此命令来更新 Experiment 的搜索空间。
  
  * 用法
    
    ```bash
    nnictl update searchspace [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写          | 是否必需  | 默认值 | 说明                    |
  | -------------- | ----- | --- | --------------------- |
  | id             | False |     | 需要设置的 Experiment 的 id |
  | --filename, -f | True  |     | 新的搜索空间文件名             |
  
  * 样例
    
    `使用 'examples/trials/mnist/search_space.json' 来更新 Experiment 的搜索空间`
    
    ```bash
    nnictl update searchspace [experiment_id] --file examples/trials/mnist/search_space.json
    ```

* **nnictl update concurrency**
  
  * 说明
    
    可以用此命令来更新 Experiment 的并发设置。
  
  * 用法
    
    ```bash
    nnictl update concurrency [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写       | 是否必需  | 默认值 | 说明                    |
  | ----------- | ----- | --- | --------------------- |
  | id          | False |     | 需要设置的 Experiment 的 id |
  | --value, -v | True  |     | 允许同时运行的 Trial 的数量     |
  
  * 样例
    
    > 更新 Experiment 的并发数量
    
    ```bash
    nnictl update concurrency [experiment_id] --value [concurrency_number]
    ```

* **nnictl update duration**
  
  * 说明
    
    可以用此命令来更新 Experiment 的运行时间。
  
  * 用法
    
    ```bash
    nnictl update duration [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写       | 是否必需  | 默认值 | 说明                                                                      |
  | ----------- | ----- | --- | ----------------------------------------------------------------------- |
  | id          | False |     | 需要设置的 Experiment 的 id                                                   |
  | --value, -v | True  |     | Experiment 持续时间如没有单位，则为秒。 后缀可以为 's' 即秒 (默认值), 'm' 即分钟, 'h' 即小时或 'd' 即天。 |
  
  * 样例
    
    > 修改 Experiment 的执行时间
    
    ```bash
    nnictl update duration [experiment_id] --value [duration]
    ```

* **nnictl update trialnum**
  
  * 说明
    
    可以用此命令来更新 Experiment 的最大 Trial 数量。
  
  * 用法
    
    ```bash
    nnictl update trialnum [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写       | 是否必需  | 默认值 | 说明                    |
  | ----------- | ----- | --- | --------------------- |
  | id          | False |     | 需要设置的 Experiment 的 id |
  | --value, -v | True  |     | 需要设置的 maxtrialnum 的数量 |
  
  * 样例
    
    > 更新 Experiment 的 Trial 数量
    
    ```bash
    nnictl update trialnum --id [experiment_id] --value [trial_num]
    ```

<a name="trial"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl trial`

* **nnictl trial ls**
  
  * 说明
    
    使用此命令来查看 Trial 的信息。
  
  * 用法
    
    ```bash
    nnictl trial ls
    ```
  
  * 选项
  
  | 参数及缩写 | 是否必需  | 默认值 | 说明                    |
  | ----- | ----- | --- | --------------------- |
  | id    | False |     | 需要设置的 Experiment 的 id |

* **nnictl trial kill**
  
  * 说明
    
    此命令用于终止 Trial。
  
  * 用法
    
    ```bash
    nnictl trial kill [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写          | 是否必需  | 默认值 | 说明                    |
  | -------------- | ----- | --- | --------------------- |
  | id             | False |     | Trial 的 Experiment ID |
  | --trial_id, -T | True  |     | 需要终止的 Trial 的 ID。     |
  
  * 样例
    
    > 结束 Trial 任务
    
    ```bash
    nnictl trial [trial_id] --experiment [experiment_id]
    ```

<a name="top"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl top`

* 说明
  
  查看正在运行的 Experiment。

* 用法
  
  ```bash
  nnictl top
  ```

* 选项
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                                   |
  | ---------- | ----- | --- | ------------------------------------ |
  | id         | False |     | 需要设置的 Experiment 的 id                |
  | --time, -t | False |     | 刷新 Experiment 状态的时间间隔，单位为秒，默认值为 3 秒。 |

<a name="experiment"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `管理 Experiment 的信息`

* **nnictl experiment show**
  
  * 说明
    
    显示 Experiment 的信息。
  
  * 用法
    
    ```bash
    nnictl experiment show
    ```
  
  * 选项
  
  | 参数及缩写 | 是否必需  | 默认值 | 说明                    |
  | ----- | ----- | --- | --------------------- |
  | id    | False |     | 需要设置的 Experiment 的 id |

* **nnictl experiment status**
  
  * 说明
    
    显示 Experiment 的状态。
  
  * 用法
    
    ```bash
    nnictl experiment status
    ```
  
  * 选项
  
  | 参数及缩写 | 是否必需  | 默认值 | 说明                    |
  | ----- | ----- | --- | --------------------- |
  | id    | False |     | 需要设置的 Experiment 的 id |

* **nnictl experiment list**
  
  * 说明
    
    显示正在运行的 Experiment 的信息
  
  * 用法
    
    ```bash
    nnictl experiment list
    ```

<a name="export"></a>

* **nnictl experiment export**
  
  * 说明
    
    使用此命令，可将 Trial 的 reward 和超参导出为 csv 文件。
  
  * 用法
    
    ```bash
    nnictl experiment export [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写  | 是否必需  | 默认值 | 说明                        |
  | ------ | ----- | --- | ------------------------- |
  | id     | False |     | Experiment ID             |
  | --file | True  |     | 文件的输出路径                   |
  | --type | True  |     | 输出文件类型，仅支持 "csv" 和 "json" |
  
  * 样例
  
  > 将 Experiment 中所有 Trial 数据导出为 JSON 格式
  
  ```bash
  nnictl experiment export [experiment_id] --file [file_path] --type json
  ```

* **nnictl experiment import**
  
  * 说明
    
    可使用此命令将以前的 Trial 超参和结果导入到 Tuner 中。 数据会传入调参算法中（即 Tuner 或 Advisor）。
  
  * 用法
    
    ```bash
    nnictl experiment import [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                       |
  | ---------- | ----- | --- | ------------------------ |
  | id         | False |     | 需要将数据导入的 Experiment 的 ID |
  | --file, -f | True  |     | 需要导入的 JSON 格式的数据文件       |
  
  * 详细说明
    
    NNI 支持导入用户的数据，确保数据格式正确。 样例如下：
    
    ```json
    [
      {"parameter": {"x": 0.5, "y": 0.9}, "value": 0.03},
      {"parameter": {"x": 0.4, "y": 0.8}, "value": 0.05},
      {"parameter": {"x": 0.3, "y": 0.7}, "value": 0.04}
    ]
    ```
    
    最顶层列表的每个元素都是一个样例。 对于内置的 Tuner 和 Advisor，每个样本至少需要两个主键：`parameter` 和 `value`。 `parameter` 必须与 Experiment 的搜索空间相匹配，`parameter` 中的所有的主键（或超参）都必须与搜索空间中的主键相匹配。 否则， Tuner 或 Advisor 可能会有无法预期的行为。 `Value` 应当遵循与 `nni.report_final_result` 的输入值一样的规则，即要么时一个数字，或者是包含 `default` 主键的 dict。 对于自定义的 Tuner 或 Advisor，根据实现的不同，此文件可以是任意的 JSON 内容（例如，`import_data`）。
    
    You also can use [nnictl experiment export](#export) to export a valid json file including previous experiment trial hyperparameters and results.
    
    Currenctly, following tuner and advisor support import data:
    
    ```yml
    builtinTunerName: TPE, Anneal, GridSearch, MetisTuner
    builtinAdvisorName: BOHB
    ```
    
    *If you want to import data to BOHB advisor, user are suggested to add "TRIAL_BUDGET" in parameter as NNI do, otherwise, BOHB will use max_budget as "TRIAL_BUDGET". Here is an example:*
    
    ```json
    [
      {"parameter": {"x": 0.5, "y": 0.9, "TRIAL_BUDGET": 27}, "value": 0.03}
    ]
    ```
  
  * Examples
    
    > import data to a running experiment
    
    ```bash
    nnictl experiment [experiment_id] -f experiment_data.json
    ```

<a name="config"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl config show`

* Description
  
  Display the current context information.

* Usage
  
  ```bash
  nnictl config show
  ```

<a name="log"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Manage log`

* **nnictl log stdout**
  
  * Description
    
    Show the stdout log content.
  
  * Usage
    
    ```bash
    nnictl log stdout [options]
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |
  | --head, -h      | False    |         | show head lines of stdout            |
  | --tail, -t      | False    |         | show tail lines of stdout            |
  | --path, -p      | False    |         | show the path of stdout file         |
  
  * Example
    
    > Show the tail of stdout log content
    
    ```bash
    nnictl log stdout [experiment_id] --tail [lines_number]
    ```

* **nnictl log stderr**
  
  * Description
    
    Show the stderr log content.
  
  * Usage
    
    ```bash
    nnictl log stderr [options]
    ```
  
  * Options
  
  | Name, shorthand | Required | 默认值 | 说明                    |
  | --------------- | -------- | --- | --------------------- |
  | id              | False    |     | 需要设置的 Experiment 的 id |
  | --head, -h      | False    |     | 显示 stderr 开始的若干行      |
  | --tail, -t      | False    |     | 显示 stderr 结尾的若干行      |
  | --path, -p      | False    |     | 显示 stderr 文件的路径       |

* **nnictl log trial**
  
  * 说明
    
    显示 Trial 日志的路径。
  
  * 用法
    
    ```bash
    nnictl log trial [options]
    ```
  
  * 选项
  
  | 参数及缩写          | 是否必需  | 默认值 | 说明                                     |
  | -------------- | ----- | --- | -------------------------------------- |
  | id             | False |     | Trial 的 Experiment ID                  |
  | --trial_id, -T | False |     | 所需要找日志路径的 Trial 的 ID，当 id 不为空时，此值也为必需。 |

<a name="webui"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `Manage webui`

* **nnictl webui url**

<a name="tensorboard"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `Manage tensorboard`

* **nnictl tensorboard start**
  
  * 说明
    
    Start the tensorboard process.
  
  * 用法
    
    ```bash
    nnictl tensorboard start
    ```
  
  * 选项
  
  | 参数及缩写          | 是否必需  | 默认值  | 说明                                   |
  | -------------- | ----- | ---- | ------------------------------------ |
  | id             | False |      | ID of the experiment you want to set |
  | --trial_id, -T | False |      | ID of the trial                      |
  | --port         | False | 6006 | The port of the tensorboard process  |
  
  * Detail
    
    1. NNICTL support tensorboard function in local and remote platform for the moment, other platforms will be supported later. 
    2. If you want to use tensorboard, you need to write your tensorboard log data to environment variable [NNI_OUTPUT_DIR] path. 
    3. In local mode, nnictl will set --logdir=[NNI_OUTPUT_DIR] directly and start a tensorboard process.
    4. In remote mode, nnictl will create a ssh client to copy log data from remote machine to local temp directory firstly, and then start a tensorboard process in your local machine. You need to notice that nnictl only copy the log data one time when you use the command, if you want to see the later result of tensorboard, you should execute nnictl tensorboard command again.
    5. If there is only one trial job, you don't need to set trial id. If there are multiple trial jobs running, you should set the trial id, or you could use [nnictl tensorboard start --trial_id all] to map --logdir to all trial log paths.

* **nnictl tensorboard stop**
  
  * 说明
    
    Stop all of the tensorboard process.
  
  * 用法
    
    ```bash
    nnictl tensorboard stop
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |

<a name="package"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Manage package`

* **nnictl package install**
  
  * 说明
    
    Install the packages needed in nni experiments.
  
  * 用法
    
    ```bash
    nnictl package install [OPTIONS]
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                         |
  | --------------- | -------- | ------- | ----------------------------------- |
  | --name          | True     |         | The name of package to be installed |
  
  * Example
    
    > Install the packages needed in tuner SMAC
    
    ```bash
    nnictl package install --name=SMAC
    ```

* **nnictl package show**
  
  * Description
    
    List the packages supported.
  
  * Usage
    
    ```bash
    nnictl package show
    ```

<a name="version"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Check NNI version`

* **nnictl --version**
  
  * Description
    
    Describe the current version of NNI installed.
  
  * Usage
    
    ```bash
    nnictl --version
    ```