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
```

### 管理实验

* **nnictl create**
  
  * 说明
    
        此命令使用参数中的配置文件，来创建新的实验。
            此命令成功完成后，上下文会被设置为此实验。这意味着如果不显式改变上下文（暂不支持），输入的以下命令，都作用于此实验。
        
  
  * 用法
    
    ```bash
    nnictl create [OPTIONS]
    ```
    
    选项：  
    
    
    | 参数及缩写        | 是否必需  | 默认值 | 说明            |
    | ------------ | ----- | --- | ------------- |
    | --config, -c | True  |     | 实验的 yaml 配置文件 |
    | --port, -p   | False |     | RESTful 服务的端口 |

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
      
      可以用此命令来更新实验的并发设置。
    
    * 用法
      
      ```bash
      nnictl update concurrency [OPTIONS]
      ```
      
      选项：
      
      | 参数及缩写       | 是否必需  | 默认值 | 说明           |
      | ----------- | ----- | --- | ------------ |
      | id          | False |     | 需要设置的实验的 id  |
      | --value, -v | True  |     | 允许同时运行的尝试的数量 |
    
    * **nnictl update duration**
      
      * 说明
        
        可以用此命令来更新实验的运行时间。
      
      * 用法
        
        ```bash
        nnictl update duration [OPTIONS]
        ```
        
        选项：
        
        | 参数及缩写       | 是否必需  | 默认值 | 说明                                                             |
        | ----------- | ----- | --- | -------------------------------------------------------------- |
        | id          | False |     | 需要设置的实验的 id                                                    |
        | --value, -v | True  |     | 实验持续时间如没有单位，则为秒。 后缀可以为 's' 即秒 (默认值), 'm' 即分钟, 'h' 即小时或 'd' 即天。 |
    
    * **nnictl update trialnum**
      
      * 说明
        
        可以用此命令来更新实验的最大尝试数量。
      
      * 用法
        
        ```bash
        nnictl update trialnum [OPTIONS]
        ```
        
        选项：
        
        | 参数及缩写       | 是否必需  | 默认值 | 说明                    |
        | ----------- | ----- | --- | --------------------- |
        | id          | False |     | 需要设置的实验的 id           |
        | --value, -v | True  |     | 需要设置的 maxtrialnum 的数量 |

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
    
    Show the information of all the (running) experiments.
  
  * Usage
    
    ```bash
    nnictl experiment list
    ```
    
    Options:
    
    | Name, shorthand | Required | Default | Description                                             |
    | --------------- | -------- | ------- | ------------------------------------------------------- |
    | all             | False    | False   | Show all of experiments, including stopped experiments. |

* **nnictl config show**
  
  * Description
    
    Display the current context information.
  
  * Usage
    
    ```bash
    nnictl config show
    ```

### Manage log

* **nnictl log stdout**
  
  * Description
    
    Show the stdout log content.
  
  * Usage
    
    ```bash
    nnictl log stdout [options]
    ```
    
    Options:
    
    | Name, shorthand | Required | Default | Description                          |
    | --------------- | -------- | ------- | ------------------------------------ |
    | id              | False    |         | ID of the experiment you want to set |
    | --head, -h      | False    |         | show head lines of stdout            |
    | --tail, -t      | False    |         | show tail lines of stdout            |
    | --path, -p      | False    |         | show the path of stdout file         |

* **nnictl log stderr**
  
  * Description
    
    Show the stderr log content.
  
  * Usage
    
    ```bash
    nnictl log stderr [options]
    ```
    
    Options:
    
    | Name, shorthand | Required | Default | Description                          |
    | --------------- | -------- | ------- | ------------------------------------ |
    | id              | False    |         | ID of the experiment you want to set |
    | --head, -h      | False    |         | show head lines of stderr            |
    | --tail, -t      | False    |         | show tail lines of stderr            |
    | --path, -p      | False    |         | show the path of stderr file         |

* **nnictl log trial**
  
  * Description
    
    Show trial log path.
  
  * Usage
    
    ```bash
    nnictl log trial [options]
    ```
    
    Options:
    
    | Name, shorthand | Required | Default | Description     |
    | --------------- | -------- | ------- | --------------- |
    | id              | False    |         | the id of trial |

### Manage webui

* **nnictl webui url**
  
  * Description
    
    Show the urls of the experiment.
  
  * Usage
    
    ```bash
    nnictl webui url
    ```
    
    Options:
    
    | Name, shorthand | Required | Default | Description                          |
    | --------------- | -------- | ------- | ------------------------------------ |
    | id              | False    |         | ID of the experiment you want to set |

### Manage tensorboard

* **nnictl tensorboard start**
  
  * Description
    
    Start the tensorboard process.
  
  * Usage
    
    ```bash
    nnictl tensorboard start
    ```
    
    Options:
    
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
    
    Options:
    
    | Name, shorthand | Required | Default | Description                          |
    | --------------- | -------- | ------- | ------------------------------------ |
    | id              | False    |         | ID of the experiment you want to set |