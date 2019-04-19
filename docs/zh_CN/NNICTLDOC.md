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
  
  | 参数及缩写            | 是否必需  | 默认值 | 说明                    |
  | ---------------- | ----- | --- | --------------------- |
  | id               | False |     | 要终止的 Trial 的 id       |
  | --experiment, -E | True  |     | Trial 的 Experiment id |
  
  * 样例
    
    > 结束 Trial 任务
    
    ```bash
    nnictl trial [trial_id] --experiment [experiment_id]
    ```

* **nnictl trial export**
  
  * 说明
    
    使用此命令，可将 Trial 的 reward 和超参导出为 csv 文件。
  
  * 用法
    
    ```bash
    nnictl trial export [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写  | 是否必需  | 默认值 | 说明            |
  | ------ | ----- | --- | ------------- |
  | id     | False |     | Experiment ID |
  | --file | True  |     | csv 文件的输出路径   |

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

<a name="config"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl config show`

* 说明
  
  显示当前上下文信息。

* 用法
  
  ```bash
  nnictl config show
  ```

<a name="log"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `管理日志`

* **nnictl log stdout**
  
  * 说明
    
    显示 stdout 日志内容。
  
  * 用法
    
    ```bash
    nnictl log stdout [options]
    ```
  
  * 选项
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                    |
  | ---------- | ----- | --- | --------------------- |
  | id         | False |     | 需要设置的 Experiment 的 id |
  | --head, -h | False |     | 显示 stdout 开始的若干行      |
  | --tail, -t | False |     | 显示 stdout 结尾的若干行      |
  | --path, -p | False |     | 显示 stdout 文件的路径       |
  
  * 样例
    
    > 显示 stdout 结尾的若干行
    
    ```bash
    nnictl log stdout [experiment_id] --tail [lines_number]
    ```

* **nnictl log stderr**
  
  * 说明
    
    显示 stderr 日志内容。
  
  * 用法
    
    ```bash
    nnictl log stderr [options]
    ```
  
  * 选项
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                    |
  | ---------- | ----- | --- | --------------------- |
  | id         | False |     | 需要设置的 Experiment 的 id |
  | --head, -h | False |     | 显示 stderr 开始的若干行      |
  | --tail, -t | False |     | 显示 stderr 结尾的若干行      |
  | --path, -p | False |     | 显示 stderr 文件的路径       |

* **nnictl log trial**
  
  * 说明
    
    显示 Trial 日志的路径。
  
  * 用法
    
    ```bash
    nnictl log trial [options]
    ```
  
  * 选项
  
  | 参数及缩写            | 是否必需  | 默认值 | 说明                            |
  | ---------------- | ----- | --- | ----------------------------- |
  | id               | False |     | 需要显示日志路径的 Trial id            |
  | --experiment, -E | False |     | 如果 id 为空，则需要添加 Experiment id。 |

<a name="webui"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `管理 Web 界面`

* **nnictl webui url**

<a name="tensorboard"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `管理 TensorBoard`

* **nnictl tensorboard start**
  
  * 说明
    
    启动 tensorboard 进程。
  
  * 用法
    
    ```bash
    nnictl tensorboard start
    ```
  
  * 选项
  
  | 参数及缩写     | 是否必需  | 默认值  | 说明                    |
  | --------- | ----- | ---- | --------------------- |
  | id        | False |      | 需要设置的 Experiment 的 id |
  | --trialid | False |      | Trial 的 id            |
  | --port    | False | 6006 | tensorboard 进程的端口     |
  
  * 详细说明
    
    1. NNICTL 当前仅支持本机和远程平台的 tensorboard，其它平台暂不支持。 
    2. 如果要使用 tensorboard，需要将 tensorboard 日志输出到环境变量 [NNI_OUTPUT_DIR] 路径下。 
    3. 在 local 模式中，nnictl 会直接设置 --logdir=[NNI_OUTPUT_DIR] 并启动 tensorboard 进程。
    4. 在 remote 模式中，nnictl 会创建一个 ssh 客户端来将日志数据从远程计算机复制到本机临时目录中，然后在本机开始 tensorboard 进程。 需要注意的是，nnictl 只在使用此命令时复制日志数据，如果要查看最新的 tensorboard 结果，需要再次执行 nnictl tensorboard 命令。
    5. 如果只有一个 Trial 任务，不需要设置 trialid。 如果有多个运行的 Trial 作业，需要设置 trialid，或使用 [nnictl tensorboard start --trialid all] 来将 --logdir 映射到所有 Trial 的路径。

* **nnictl tensorboard stop**
  
  * 说明
    
    停止所有 tensorboard 进程。
  
  * 用法
    
    ```bash
    nnictl tensorboard stop
    ```
  
  * 选项
  
  | 参数及缩写 | 是否必需  | 默认值 | 说明                    |
  | ----- | ----- | --- | --------------------- |
  | id    | False |     | 需要设置的 Experiment 的 id |

<a name="package"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `管理包`

* **nnictl package install**
  
  * 说明
    
    安装 NNI 实验所需要的包。
  
  * 用法
    
    ```bash
    nnictl package install [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写  | 是否必需 | 默认值 | 说明      |
  | ------ | ---- | --- | ------- |
  | --name | True |     | 要安装的包名称 |
  
  * 样例
    
    > 安装 SMAC Tuner 所需要的包
    
    ```bash
    nnictl package install --name=SMAC
    ```

* **nnictl package show**
  
  * 说明
    
    列出支持的安装包
  
  * 用法
    
    ```bash
    nnictl package show
    ```

<a name="version"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `检查 NNI 版本`

* **nnictl --version**
  
  * 说明
    
    显示当前安装的 NNI 的版本。
  
  * 用法
    
    ```bash
    nnictl --version
    ```