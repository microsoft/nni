nnictl

===

## 介绍

**nnictl** 是一个命令行工具，用来控制 NNI 实验，如启动、停止、继续实验，启动、停止 NNIBoard 等等。

## 命令

nnictl 支持的命令：

```bash
nnictl create
nnictl stop
nnictl update
nnictl resume
nnictl trial
nnictl experiment
nnictl config
nnictl log
nnictl webui
nnictl tensorboard
nnictl top
nnictl --version
```

### 管理实验

* **nnictl create**
  
  * 说明
    
    此命令使用参数中的配置文件，来创建新的实验。 此命令成功完成后，上下文会被设置为此实验。这意味着如果不显式改变上下文（暂不支持），输入的以下命令，都作用于此实验。
  
  * 用法
    
    ```bash
    nnictl create [OPTIONS]
    ```
    
    选项：  
    
    
    | 参数及缩写        | 是否必需  | 默认值 | 说明                                    |
    | ------------ | ----- | --- | ------------------------------------- |
    | --config, -c | True  |     | yaml configure file of the experiment |
    | --port, -p   | False |     | the port of restful server            |

* **nnictl resume**
  
  * 说明
    
    使用此命令恢复已停止的实验。
  
  * 用法
    
    ```bash
    nnictl resume [OPTIONS]
    ```
    
    选项：
    
    | 参数及缩写      | 是否必需  | 默认值 | 说明                     |
    | ---------- | ----- | --- | ---------------------- |
    | id         | False |     | 要恢复的实验标识               |
    | --port, -p | False |     | 要恢复的实验使用的 RESTful 服务端口 |

* **nnictl stop**
  
  * 说明
    
    使用此命令来停止正在运行的单个或多个实验。
  
  * 用法
    
    ```bash
    nnictl stop [id]
    ```
  
  * 详细说明
    
    1. 如果指定了 id，并且此 id 匹配正在运行的实验，nnictl 会停止相应的实验，否则会输出错误信息。
    
    2. 如果没有指定 id，并且当前有运行的实验，则会停止该实验，否则会输出错误信息。
    
    3. 如果 id 以 * 结尾，nnictl 会停止所有匹配此通配符的实验。
    
    4. 如果 id 不存在，但匹配了某个实验的 id 前缀，nnictl 会停止匹配的实验。
    
    5. 如果 id 不存在，但匹配多个实验 id 的前缀，nnictl 会输出这些 id 的信息。
    
    6. 可使用 'nnictl stop all' 来停止所有的实验。

* **nnictl update**
  
  * **nnictl update searchspace**
    
    * 说明
      
      可以用此命令来更新实验的搜索空间。
    
    * 用法
      
      ```bash
      nnictl update searchspace [OPTIONS]
      ```
      
      选项：
      
      | 参数及缩写          | 是否必需  | 默认值 | 说明          |
      | -------------- | ----- | --- | ----------- |
      | id             | False |     | 需要设置的实验的 id |
      | --filename, -f | True  |     | 新的搜索空间文件名   |
  
  * **nnictl update concurrency**
    
    * 说明
      
      You can use this command to update an experiment's concurrency.
    
    * 用法
      
      ```bash
      nnictl update concurrency [OPTIONS]
      ```
      
      选项：
      
      | 参数及缩写       | 是否必需  | 默认值 | 说明                                      |
      | ----------- | ----- | --- | --------------------------------------- |
      | id          | False |     | 需要设置的实验的 id                             |
      | --value, -v | True  |     | the number of allowed concurrent trials |
  
  * **nnictl update duration**
    
    * 说明
      
      可以用此命令来更新实验的运行时间。
    
    * 用法
      
      ```bash
      nnictl update duration [OPTIONS]
      ```
      
      选项：
      
      | 参数及缩写       | 是否必需  | 默认值 | 说明                                                                                                                                           |
      | ----------- | ----- | --- | -------------------------------------------------------------------------------------------------------------------------------------------- |
      | id          | False |     | 需要设置的实验的 id                                                                                                                                  |
      | --value, -v | True  |     | the experiment duration will be NUMBER seconds. SUFFIX may be 's' for seconds (the default), 'm' for minutes, 'h' for hours or 'd' for days. |
  
  * **nnictl update trialnum**
    
    * 说明
      
      You can use this command to update an experiment's maxtrialnum.
    
    * 用法
      
      ```bash
      nnictl update trialnum [OPTIONS]
      ```
      
      选项：
      
      | 参数及缩写       | 是否必需  | 默认值 | 说明                                            |
      | ----------- | ----- | --- | --------------------------------------------- |
      | id          | False |     | 需要设置的实验的 id                                   |
      | --value, -v | True  |     | the new number of maxtrialnum you want to set |

* **nnictl trial**
  
  * **nnictl trial ls**
    
    * 说明
      
      使用此命令来查看尝试的信息。
    
    * 用法
      
      ```bash
      nnictl trial ls
      ```
      
      选项：
      
      | 参数及缩写 | 是否必需  | 默认值 | 说明          |
      | ----- | ----- | --- | ----------- |
      | id    | False |     | 需要设置的实验的 id |
  
  * **nnictl trial kill**
    
    * 说明
      
      此命令用于终止尝试。
    
    * 用法
      
      ```bash
      nnictl trial kill [OPTIONS]
      ```
      
      选项：
      
      | 参数及缩写         | 是否必需  | 默认值 | 说明           |
      | ------------- | ----- | --- | ------------ |
      | id            | False |     | 需要设置的实验的 id  |
      | --trialid, -t | True  |     | 需要终止的尝试的 id。 |
  
  * **nnictl top**
    
    * 说明
      
      查看正在运行的实验。
    
    * 用法
      
      ```bash
      nnictl top
      ```
      
      选项：
      
      | 参数及缩写      | 是否必需  | 默认值 | 说明                         |
      | ---------- | ----- | --- | -------------------------- |
      | id         | False |     | 需要设置的实验的 id                |
      | --time, -t | False |     | 刷新实验状态的时间间隔，单位为秒，默认值为 3 秒。 |

### 管理实验信息

* **nnictl experiment show**
  
  * 说明
    
    显示实验的信息。
  
  * 用法
    
    ```bash
    nnictl experiment show
    ```
    
    选项：
    
    | 参数及缩写 | 是否必需  | 默认值 | 说明          |
    | ----- | ----- | --- | ----------- |
    | id    | False |     | 需要设置的实验的 id |

* **nnictl experiment status**
  
  * 说明
    
    显示实验的状态。
  
  * 用法
    
    ```bash
    nnictl experiment status
    ```
    
    选项：
    
    | 参数及缩写 | 是否必需  | 默认值 | 说明          |
    | ----- | ----- | --- | ----------- |
    | id    | False |     | 需要设置的实验的 id |

* **nnictl experiment list**
  
  * 说明
    
    显示正在运行的实验的信息
  
  * 用法
    
    ```bash
    nnictl experiment list
    ```
    
    选项：
    
    | 参数及缩写 | 是否必需  | 默认值   | 说明               |
    | ----- | ----- | ----- | ---------------- |
    | all   | False | False | 显示所有实验，包括已停止的实验。 |

* **nnictl config show**
  
  * 说明
    
    Display the current context information.
  
  * 用法
    
    ```bash
    nnictl config show
    ```

### 管理日志

* **nnictl log stdout**
  
  * 说明
    
    显示 stdout 日志内容。
  
  * 用法
    
    ```bash
    nnictl log stdout [options]
    ```
    
    选项：
    
    | 参数及缩写      | 是否必需  | 默认值 | 说明               |
    | ---------- | ----- | --- | ---------------- |
    | id         | False |     | 需要设置的实验的 id      |
    | --head, -h | False |     | 显示 stdout 开始的若干行 |
    | --tail, -t | False |     | 显示 stdout 结尾的若干行 |
    | --path, -p | False |     | 显示 stdout 文件的路径  |

* **nnictl log stderr**
  
  * 说明
    
    显示 stderr 日志内容。
  
  * 用法
    
    ```bash
    nnictl log stderr [options]
    ```
    
    选项：
    
    | 参数及缩写      | 是否必需  | 默认值 | 说明               |
    | ---------- | ----- | --- | ---------------- |
    | id         | False |     | 需要设置的实验的 id      |
    | --head, -h | False |     | 显示 stderr 开始的若干行 |
    | --tail, -t | False |     | 显示 stderr 结尾的若干行 |
    | --path, -p | False |     | 显示 stderr 文件的路径  |

* **nnictl log trial**
  
  * 说明
    
    显示尝试日志的路径。
  
  * 用法
    
    ```bash
    nnictl log trial [options]
    ```
    
    选项：
    
    | 参数及缩写 | 是否必需  | 默认值 | 说明     |
    | ----- | ----- | --- | ------ |
    | id    | False |     | 尝试的 id |

### 管理网页

* **nnictl webui url**
  
  * 说明
    
    显示实验的 URL。
  
  * 用法
    
    ```bash
    nnictl webui url
    ```
    
    选项：
    
    | 参数及缩写 | 是否必需  | 默认值 | 说明          |
    | ----- | ----- | --- | ----------- |
    | id    | False |     | 需要设置的实验的 id |

### 管理 tensorboard

* **nnictl tensorboard start**
  
  * 说明
    
    启动 tensorboard 进程。
  
  * 用法
    
    ```bash
    nnictl tensorboard start
    ```
    
    选项：
    
    | 参数及缩写     | 是否必需  | 默认值  | 说明                |
    | --------- | ----- | ---- | ----------------- |
    | id        | False |      | 需要设置的实验的 id       |
    | --trialid | False |      | 尝试的 id            |
    | --port    | False | 6006 | tensorboard 进程的端口 |
  
  * 详细说明
    
    1. NNICTL support tensorboard function in local and remote platform for the moment, other platforms will be supported later. 
    2. If you want to use tensorboard, you need to write your tensorboard log data to environment variable [NNI_OUTPUT_DIR] path. 
    3. In local mode, nnictl will set --logdir=[NNI_OUTPUT_DIR] directly and start a tensorboard process.
    4. In remote mode, nnictl will create a ssh client to copy log data from remote machine to local temp directory firstly, and then start a tensorboard process in your local machine. You need to notice that nnictl only copy the log data one time when you use the command, if you want to see the later result of tensorboard, you should execute nnictl tensorboard command again.
    5. If there is only one trial job, you don't need to set trialid. If there are multiple trial jobs running, you should set the trialid, or you could use [nnictl tensorboard start --trialid all] to map --logdir to all trial log paths.

* **nnictl tensorboard stop**
  
  * 说明
    
    停止所有 tensorboard 进程。
  
  * 用法
    
    ```bash
    nnictl tensorboard stop
    ```
    
    选项：
    
    | 参数及缩写 | 是否必需  | 默认值 | 说明          |
    | ----- | ----- | --- | ----------- |
    | id    | False |     | 需要设置的实验的 id |

### 检查 NNI 版本

* **nnictl --version**
  
  * 说明
    
    显示当前安装的 NNI 的版本。
  
  * 用法
    
    ```bash
    nnictl --version
    ```