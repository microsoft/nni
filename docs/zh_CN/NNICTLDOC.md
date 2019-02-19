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
    5. 如果 id 不存在，但匹配了多个 Experiment id 的前缀，nnictl 会输出这些 id 的信息。
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
  
  | 参数及缩写          | Required | Default | Description                            |
  | -------------- | -------- | ------- | -------------------------------------- |
  | id             | False    |         | ID of the experiment you want to set   |
  | --filename, -f | True     |         | the file storing your new search space |
  
  * **nnictl update concurrency**
    
    * Description
      
      You can use this command to update an experiment's concurrency.
    
    * Usage
      
      ```bash
      nnictl update concurrency [OPTIONS]
      ```
    
    * Options
  
  | Name, shorthand | Required | Default | Description                             |
  | --------------- | -------- | ------- | --------------------------------------- |
  | id              | False    |         | ID of the experiment you want to set    |
  | --value, -v     | True     |         | the number of allowed concurrent trials |
  
  * **nnictl update duration**
    
    * Description
      
      You can use this command to update an experiment's duration.
    
    * Usage
      
      ```bash
      nnictl update duration [OPTIONS]
      ```
    
    * Options
  | Name, shorthand | Required | Default | Description                                                                                                                                  |
  | --------------- | -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
  | id              | False    |         | ID of the experiment you want to set                                                                                                         |
  | --value, -v     | True     |         | the experiment duration will be NUMBER seconds. SUFFIX may be 's' for seconds (the default), 'm' for minutes, 'h' for hours or 'd' for days. |
  
  * **nnictl update trialnum**
    
    * Description
      
      You can use this command to update an experiment's maxtrialnum.
    
    * Usage
      
      ```bash
      nnictl update trialnum [OPTIONS]
      ```
    
    * Options
  
  | Name, shorthand | Required | Default | Description                                   |
  | --------------- | -------- | ------- | --------------------------------------------- |
  | id              | False    |         | ID of the experiment you want to set          |
  | --value, -v     | True     |         | the new number of maxtrialnum you want to set |

<a name="trial"></a>

* **nnictl trial**
  
  * **nnictl trial ls**
    
    * Description
      
      You can use this command to show trial's information.
    
    * Usage
      
      ```bash
      nnictl trial ls
      ```
    
    * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |
  
  * **nnictl trial kill**
    
    * Description
      
      You can use this command to kill a trial job.
    
    * Usage
      
      ```bash
      nnictl trial kill [OPTIONS]
      ```
    
    * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |
  | --trialid, -t   | True     |         | ID of the trial you want to kill.    |

<a name="top"></a>

* **nnictl top**
  
  * Description
    
    Monitor all of running experiments.
  
  * Usage
    
    ```bash
    nnictl top
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                                                                                                  |
  | --------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set                                                                         |
  | --time, -t      | False    |         | The interval to update the experiment status, the unit of time is second, and the default value is 3 second. |

<a name="experiment"></a>

### Manage experiment information

* **nnictl experiment show**
  
  * Description
    
    Show the information of experiment.
  
  * Usage
    
    ```bash
    nnictl experiment show
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |

* **nnictl experiment status**
  
  * Description
    
    Show the status of experiment.
  
  * Usage
    
    ```bash
    nnictl experiment status
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                          |
  | --------------- | -------- | ------- | ------------------------------------ |
  | id              | False    |         | ID of the experiment you want to set |

* **nnictl experiment list**
  
  * Description
    
    Show the information of all the (running) experiments.
  
  * Usage
    
    ```bash
    nnictl experiment list
    ```

<a name="config"></a>

* **nnictl config show**
  
  * Description
    
    Display the current context information.
  
  * Usage
    
    ```bash
    nnictl config show
    ```

<a name="log"></a>

### Manage log

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

* **nnictl log stderr**
  
  * Description
    
    Show the stderr log content.
  
  * Usage
    
    ```bash
    nnictl log stderr [options]
    ```
  
  * Options
  
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
  
  * Options
  
  | Name, shorthand | Required | Default | Description     |
  | --------------- | -------- | ------- | --------------- |
  | id              | False    |         | the id of trial |

<a name="webui"></a>

### Manage webui

* **nnictl webui url**

<a name="tensorboard"></a>

### Manage tensorboard

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