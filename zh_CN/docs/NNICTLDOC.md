# nnictl

## 介绍

**nnictl** is a command line tool, which can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc.

## 命令

nnictl support commands:

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
    

### 管理实验

* **nnictl create**
   
   * 说明
      
      此命令使用参数中的配置文件，来创建新的实验。 After this command is successfully done, the context will be set as this experiment, which means the following command you issued is associated with this experiment, unless you explicitly changes the context(not supported yet).
   
   * 用法
      
      nnictl create [OPTIONS]
      
      Options:
      
      | Name, shorthand | Required | Default | Description                           |
      | --------------- | -------- | ------- | ------------------------------------- |
      | --config, -c    | True     |         | yaml configure file of the experiment |
      | --port, -p      | False    |         | the port of restful server            |

* **nnictl resume**
   
   * 说明
      
      使用此命令恢复已停止的实验。
   
   * 用法
      
      nnictl resume [OPTIONS]  
      Options:
      
      | 参数及缩写      | 是否必需  | 默认值 | 说明                     |
      | ---------- | ----- | --- | ---------------------- |
      | id         | False |     | 要恢复的实验标识               |
      | --port, -p | False |     | 要恢复的实验使用的 RESTful 服务端口 |

* **nnictl stop**
   
   * 说明
      
      使用此命令来停止正在运行的单个或多个实验。
   
   * 用法
      
      nnictl stop [id]
   
   * 详细说明
      
      1.If there is an id specified, and the id matches the running experiment, nnictl will stop the corresponding experiment, or will print error message. 2.If there is no id specified, and there is an experiment running, stop the running experiment, or print error message. 3.If the id ends with *, nnictl will stop all experiments whose ids matchs the regular. 4.If the id does not exist but match the prefix of an experiment id, nnictl will stop the matched experiment. 5.If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information. 6.Users could use 'nnictl stop all' to stop all experiments

* **nnictl update**
   
   * **nnictl update searchspace**
      
      * 说明
         
         可以用此命令来更新实验的搜索空间。
      
      * 用法
         
               nnictl update searchspace [OPTIONS] 
             
             Options:
             
         
         | 参数及缩写          | 是否必需  | 默认值 | 说明          |
         | -------------- | ----- | --- | ----------- |
         | id             | False |     | 需要设置的实验的 id |
         | --filename, -f | True  |     | 新的搜索空间文件名   |
         
         * **nnictl update concurrency** 
         * Description
            
            You can use this command to update an experiment's concurrency.
         
         * Usage
            
            nnictl update concurrency [OPTIONS]
            
            Options:
            
            | Name, shorthand | Required | Default | Description                             |
            | --------------- | -------- | ------- | --------------------------------------- |
            | id              | False    |         | ID of the experiment you want to set    |
            | --value, -v     | True     |         | the number of allowed concurrent trials |
         
         * **nnictl update duration**
         
         * Description
            
                You can use this command to update an experiment's concurrency.  
                
         
         * Usage
            
            nnictl update duration [OPTIONS]
            
            Options:
         
         | Name, shorthand | Required | Default | Description                                                                                                                                  |
         | --------------- | -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
         | id              | False    |         | ID of the experiment you want to set                                                                                                         |
         | --value, -v     | True     |         | the experiment duration will be NUMBER seconds. SUFFIX may be 's' for seconds (the default), 'm' for minutes, 'h' for hours or 'd' for days. |
         
         * **nnictl update trialnum** 
         * Description
            
            You can use this command to update an experiment's maxtrialnum.
         
         * Usage
            
            nnictl update trialnum [OPTIONS]
            
            Options:
            
            | Name, shorthand | Required | Default | Description                                   |
            | --------------- | -------- | ------- | --------------------------------------------- |
            | id              | False    |         | ID of the experiment you want to set          |
            | --value, -v     | True     |         | the new number of maxtrialnum you want to set |

* **nnictl trial**
   
   * **nnictl trial ls**
      
      * 说明
         
         使用此命令来查看尝试的信息。
      
      * 用法
         
         nnictl trial ls
      
      Options:
      
      | Name, shorthand | Required | Default | Description                          |
      | --------------- | -------- | ------- | ------------------------------------ |
      | id              | False    |         | ID of the experiment you want to set |
   
   * **nnictl trial kill**
      
      * 说明
         
         此命令用于终止尝试。
      
      * 用法
         
               nnictl trial kill [OPTIONS] 
             
         
         选项：
         
         | 参数及缩写         | 是否必需  | 默认值 | 说明           |
         | ------------- | ----- | --- | ------------ |
         | id            | False |     | 需要设置的实验的 id  |
         | --trialid, -t | True  |     | 需要终止的尝试的 id。 |
   
   * **nnictl top**
      
      * 说明
         
         查看正在运行的实验。
      
      * 用法
         
               nnictl top
             
         
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
      
      nnictl experiment show
      
      Options:
      
      | 参数及缩写 | 是否必需  | 默认值 | 说明          |
      | ----- | ----- | --- | ----------- |
      | id    | False |     | 需要设置的实验的 id |

* **nnictl experiment status**
   
   * 说明
      
      显示实验的状态。
   
   * 用法
      
      nnictl experiment status
      
      Options:
      
      | 参数及缩写 | 是否必需  | 默认值 | 说明          |
      | ----- | ----- | --- | ----------- |
      | id    | False |     | 需要设置的实验的 id |

* **nnictl experiment list**
   
   * 说明
      
      显示正在运行的实验的信息
   
   * 用法
      
      nnictl experiment list
      
      Options:
      
      | 参数及缩写 | 是否必需  | 默认值   | 说明               |
      | ----- | ----- | ----- | ---------------- |
      | all   | False | False | 显示所有实验，包括已停止的实验。 |

* **nnictl config show**
   
   * 说明
      
           Display the current context information.
          
   
   * 用法
      
          nnictl config show
          

### 管理日志

* **nnictl log stdout**
   
   * 说明
      
      显示 stdout 日志内容。
   
   * 用法
      
           nnictl log stdout [options]
          
      
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
      
      nnictl log stderr [options]
      
      Options:
      
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
      
      nnictl log trial [options]
      
      Options:
      
      | 参数及缩写 | 是否必需  | 默认值 | 说明     |
      | ----- | ----- | --- | ------ |
      | id    | False |     | 尝试的 id |

### 管理网页

* **nnictl webui url**
   
   * 说明
      
      显示实验的 URL。
   
   * 用法
      
           nnictl webui url
          
      
      选项：
      
      | 参数及缩写 | 是否必需  | 默认值 | 说明          |
      | ----- | ----- | --- | ----------- |
      | id    | False |     | 需要设置的实验的 id |

### 管理 tensorboard

* **nnictl tensorboard start**
   
   * 说明
      
      启动 tensorboard 进程。
   
   * 用法
      
           nnictl tensorboard start
          
      
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
      
           nnictl tensorboard stop
          
      
      选项：
      
      | 参数及缩写 | 是否必需  | 默认值 | 说明          |
      | ----- | ----- | --- | ----------- |
      | id    | False |     | 需要设置的实验的 id |