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
* [nnictl experiment show](#experiment)
* [nnictl config](#config)
* [nnictl log](#log)
* [nnictl webui](#webui)
* [nnictl tensorboard](#tensorboard)
* [nnictl package](#package)
* [nnictl --version](#version)

### 管理 Experiment

<a name="create"></a>

* **nnictl create**
  
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

<a name="resume"></a>

* **nnictl resume**
  
  * 说明
    
    使用此命令恢复已停止的 Experiment。
  
  * 用法
    
    ```bash
    nnictl resume [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                               |
  | ---------- | ----- | --- | -------------------------------- |
  | id         | False |     | 要恢复的 Experiment 标识               |
  | --port, -p | False |     | 要恢复的 Experiment 使用的 RESTful 服务端口 |

<a name="stop"></a>

* **nnictl stop**
  
  * 说明
    
    使用此命令来停止正在运行的单个或多个 Experiment。
  
  * 用法
    
    ```bash
    nnictl stop [id]
    ```
  
  * 详细说明
    
    1. 如果指定了 id，并且此 id 匹配正在运行的 Experiment，nnictl 会停止相应的 Experiment，否则会输出错误信息。
    2. 如果没有指定 id，并且当前有运行的 Experiment，则会停止该 Experiment，否则会输出错误信息。
    3. 如果 id 以 * 结尾，nnictl 会停止所有匹配此通配符的 Experiment。
    4. 如果 id 不存在，但匹配了某个Experiment 的 id 前缀，nnictl 会停止匹配的Experiment 。
    5. 如果 id 不存在，但匹配多个 Experiment id 的前缀，nnictl 会输出这些 id 的信息。
    6. 可使用 'nnictl stop all' 来停止所有的 Experiment。

<a name="update"></a>

* **nnictl update**
  
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

<a name="trial"></a>

* **nnictl trial**
  
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
  
  | 参数及缩写         | 是否必需  | 默认值 | 说明                    |
  | ------------- | ----- | --- | --------------------- |
  | id            | False |     | 需要设置的 Experiment 的 id |
  | --trialid, -t | True  |     | 需要终止的 Trial 的 id。     |

<a name="top"></a>

* **nnictl top**
  
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

### 管理 Experiment 信息

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

* **nnictl config show**
  
  * 说明
    
    显示当前上下文信息。
  
  * 用法
    
    ```bash
    nnictl config show
    ```

<a name="log"></a>

### 管理日志

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
  
  | 参数及缩写 | Required | Default | Description     |
  | ----- | -------- | ------- | --------------- |
  | id    | False    |         | the id of trial |

<a name="webui"></a>

### 管理网页

* **nnictl webui url**

<a name="tensorboard"></a>

### 管理 tensorboard

* **nnictl tensorboard start**
  
  * Description
    
    Start the tensorboard process.
  
  * Usage
    
    ```bash
    nnictl tensorboard start
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |
  | --trialid       | False    |         | ID of the trial                      |
  | --port          | False    | 6006    | The port of the tensorboard process  |
  
  * Detail
    
    1. NNICTL support tensorboard function in local and remote platform for the moment, other platforms will be supported later. 
    2. If you want to use tensorboard, you need to write your tensorboard log data to environment variable [NNI_OUTPUT_DIR] path. 
    3. In local mode, nnictl will set --logdir=[NNI_OUTPUT_DIR] directly and start a tensorboard process.
    4. In remote mode, nnictl will create a ssh client to copy log data from remote machine to local temp directory firstly, and then start a tensorboard process in your local machine. You need to notice that nnictl only copy the log data one time when you use the command, if you want to see the later result of tensorboard, you should execute nnictl tensorboard command again.
    5. If there is only one trial job, you don't need to set trialid. If there are multiple trial jobs running, you should set the trialid, or you could use [nnictl tensorboard start --trialid all] to map --logdir to all trial log paths.

* **nnictl tensorboard stop**
  
  * Description
    
    Stop all of the tensorboard process.
  
  * Usage
    
    ```bash
    nnictl tensorboard stop
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |

<a name="package"></a>

### Manage package

* **nnictl package install**
  
  * Description
    
    Install the packages needed in nni experiments.
  
  * Usage
    
    ```bash
    nnictl package install [OPTIONS]
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                         |
  | --------------- | -------- | ------- | ----------------------------------- |
  | --name          | True     |         | The name of package to be installed |

* **nnictl package show**
  
  * Description
    
    List the packages supported.
  
  * Usage
    
    ```bash
    nnictl package show
    ```

<a name="version"></a>

### Check NNI version

* **nnictl --version**
  
  * Description
    
    Describe the current version of NNI installed.
  
  * Usage
    
    ```bash
    nnictl --version
    ```