# nnictl

## 介绍

**nnictl** 是一个命令行工具，用来控制 NNI Experiment，如启动、停止、继续 Experiment，启动、停止 NNIBoard 等等。

## 命令

nnictl 支持的命令：

* [nnictl create](#create)
* [nnictl resume](#resume)
* [nnictl view](#view)
* [nnictl stop](#stop)
* [nnictl update](#update)
* [nnictl trial](#trial)
* [nnictl top](#top)
* [nnictl experiment](#experiment)
* [nnictl platform](#platform)
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

<a name="view"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl view`

* 说明
  
  You can use this command to view a stopped experiment.

* 用法
  
  ```bash
  nnictl view [OPTIONS]
  ```

* 选项
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                                           |
  | ---------- | ----- | --- | -------------------------------------------- |
  | id         | True  |     | The id of the experiment you want to view    |
  | --port, -p | False |     | Rest port of the experiment you want to view |

* Example
  
  > view an experiment with specified port 8088
  
  ```bash
  nnictl view [experiment_id] --port 8088
  ```

<a name="stop"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl stop`

* Description
  
  You can use this command to stop a running experiment or multiple experiments.

* Usage
  
  ```bash
  nnictl stop [Options]
  ```

* Options
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                                           |
  | ---------- | ----- | --- | -------------------------------------------- |
  | id         | False |     | The id of the experiment you want to stop    |
  | --port, -p | False |     | Rest port of the experiment you want to stop |
  | --all, -a  | False |     | Stop all of experiments                      |

* Details & Examples
  
  1. If there is no id specified, and there is an experiment running, stop the running experiment, or print error message.
    
        ```bash
        nnictl stop
        ```
        
  
  2. If there is an id specified, and the id matches the running experiment, nnictl will stop the corresponding experiment, or will print error message.
    
        ```bash
        nnictl stop [experiment_id]
        ```
        
  
  3. If there is a port specified, and an experiment is running on that port, the experiment will be stopped.
    
        ```bash
        nnictl stop --port 8080
        ```
        
  
  4. Users could use 'nnictl stop --all' to stop all experiments.
    
        ```bash
        nnictl stop --all
        ```
        
  
  5. If the id ends with *, nnictl will stop all experiments whose ids matchs the regular.
  
  6. If the id does not exist but match the prefix of an experiment id, nnictl will stop the matched experiment.
  7. If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information.

<a name="update"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl update`

* **nnictl update searchspace**
  
  * 说明
    
    You can use this command to update an experiment's search space.
  
  * 用法
    
    ```bash
    nnictl update searchspace [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写          | 是否必需  | 默认值 | 说明                                     |
  | -------------- | ----- | --- | -------------------------------------- |
  | id             | False |     | 需要设置的 Experiment 的 id                  |
  | --filename, -f | True  |     | the file storing your new search space |
  
  * Example
    
    `update experiment's new search space with file dir 'examples/trials/mnist/search_space.json'`
    
    ```bash
    nnictl update searchspace [experiment_id] --filename examples/trials/mnist/search_space.json
    ```

* **nnictl update concurrency**
  
  * 说明
    
    You can use this command to update an experiment's concurrency.
  
  * 用法
    
    ```bash
    nnictl update concurrency [OPTIONS]
    ```
  
  * 选项
  
  | 参数及缩写       | 是否必需  | 默认值 | 说明                                      |
  | ----------- | ----- | --- | --------------------------------------- |
  | id          | False |     | ID of the experiment you want to set    |
  | --value, -v | True  |     | the number of allowed concurrent trials |
  
  * 样例
    
    > update experiment's concurrency
    
    ```bash
    nnictl update concurrency [experiment_id] --value [concurrency_number]
    ```

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
  
  * Example
    
    > update experiment's duration
    
    ```bash
    nnictl update duration [experiment_id] --value [duration]
    ```

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
  
  * Example
    
    > update experiment's trial num
    
    ```bash
    nnictl update trialnum --id [experiment_id] --value [trial_num]
    ```

<a name="trial"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl trial`

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
  
  | Name, shorthand | Required | Default | Description                       |
  | --------------- | -------- | ------- | --------------------------------- |
  | id              | False    |         | Experiment ID of the trial        |
  | --trial_id, -T  | True     |         | ID of the trial you want to kill. |
  
  * Example
    
    > kill trail job
    
    ```bash
    nnictl trial [trial_id] --experiment [experiment_id]
    ```

<a name="top"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl top`

* Description
  
  Monitor all of running experiments.

* Usage
  
  ```bash
  nnictl top
  ```

* Options
  
  | 参数及缩写      | 是否必需  | 默认值 | 说明                                                                                                           |
  | ---------- | ----- | --- | ------------------------------------------------------------------------------------------------------------ |
  | id         | False |     | ID of the experiment you want to set                                                                         |
  | --time, -t | False |     | The interval to update the experiment status, the unit of time is second, and the default value is 3 second. |

<a name="experiment"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `Manage experiment information`

* **nnictl experiment show**
  
  * 说明
    
    Show the information of experiment.
  
  * 用法
    
    ```bash
    nnictl experiment show
    ```
  
  * 选项
  
  | 参数及缩写 | 是否必需  | 默认值 | 说明                                   |
  | ----- | ----- | --- | ------------------------------------ |
  | id    | False |     | ID of the experiment you want to set |

* **nnictl experiment status**
  
  * 说明
    
    Show the status of experiment.
  
  * 用法
    
    ```bash
    nnictl experiment status
    ```
  
  * 选项
  
  | 参数及缩写 | 是否必需  | 默认值 | 说明                                   |
  | ----- | ----- | --- | ------------------------------------ |
  | id    | False |     | ID of the experiment you want to set |

* **nnictl experiment list**
  
  * Description
    
    Show the information of all the (running) experiments.
  
  * Usage
    
    ```bash
    nnictl experiment list [OPTIONS]
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description             |
  | --------------- | -------- | ------- | ----------------------- |
  | --all           | False    |         | list all of experiments |

* **nnictl experiment delete**
  
  * Description
    
    Delete one or all experiments, it includes log, result, environment information and cache. It uses to delete useless experiment result, or save disk space.
  
  * Usage
    
    ```bash
    nnictl experiment delete [OPTIONS]
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description               |
  | --------------- | -------- | ------- | ------------------------- |
  | id              | False    |         | ID of the experiment      |
  | --all           | False    |         | delete all of experiments |

<a name="export"></a>

* **nnictl experiment export**
  
  * 说明
    
    You can use this command to export reward & hyper-parameter of trial jobs to a csv file.
  
  * Usage
    
    ```bash
    nnictl experiment export [OPTIONS]
    ```
  
  * Options
  
  | 参数及缩写          | 是否必需  | 默认值 | 说明                                                 |
  | -------------- | ----- | --- | -------------------------------------------------- |
  | id             | False |     | ID of the experiment                               |
  | --filename, -f | True  |     | File path of the output file                       |
  | --type         | True  |     | Type of output file, only support "csv" and "json" |
  
  * Examples
  
  > export all trial data in an experiment as json format
  
  ```bash
  nnictl experiment export [experiment_id] --filename [file_path] --type json
  ```

* **nnictl experiment import**
  
  * Description
    
    You can use this command to import several prior or supplementary trial hyperparameters & results for NNI hyperparameter tuning. The data are fed to the tuning algorithm (e.g., tuner or advisor).
  
  * Usage
    
    ```bash
    nnictl experiment import [OPTIONS]
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                                           |
  | --------------- | -------- | ------- | ----------------------------------------------------- |
  | id              | False    |         | The id of the experiment you want to import data into |
  | --filename, -f  | True     |         | a file with data you want to import in json format    |
  
  * Details
    
    NNI supports users to import their own data, please express the data in the correct format. An example is shown below:
    
    ```json
    [
      {"parameter": {"x": 0.5, "y": 0.9}, "value": 0.03},
      {"parameter": {"x": 0.4, "y": 0.8}, "value": 0.05},
      {"parameter": {"x": 0.3, "y": 0.7}, "value": 0.04}
    ]
    ```
    
    Every element in the top level list is a sample. For our built-in tuners/advisors, each sample should have at least two keys: `parameter` and `value`. The `parameter` must match this experiment's search space, that is, all the keys (or hyperparameters) in `parameter` must match the keys in the search space. Otherwise, tuner/advisor may have unpredictable behavior. `Value` should follow the same rule of the input in `nni.report_final_result`, that is, either a number or a dict with a key named `default`. For your customized tuner/advisor, the file could have any json content depending on how you implement the corresponding methods (e.g., `import_data`).
    
    You also can use [nnictl experiment export](#export) to export a valid json file including previous experiment trial hyperparameters and results.
    
    Currently, following tuner and advisor support import data:
    
    ```yaml
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
    nnictl experiment import [experiment_id] -f experiment_data.json
    ```

<a name="platform"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `Manage platform information`

* **nnictl platform clean**
  
  * Description
    
    It uses to clean up disk on a target platform. The provided YAML file includes the information of target platform, and it follows the same schema as the NNI configuration file.
  
  * Note
    
    if the target platform is being used by other users, it may cause unexpected errors to others.
  
  * Usage
    
    ```bash
    nnictl platform clean [OPTIONS]
    ```
  
  * Options
  
  | Name, shorthand | Required | Default | Description                                                 |
  | --------------- | -------- | ------- | ----------------------------------------------------------- |
  | --config        | True     |         | the path of yaml config file used when create an experiment |

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
  
  | Name, shorthand | Required | Default | Description                                                              |
  | --------------- | -------- | ------- | ------------------------------------------------------------------------ |
  | id              | False    |         | Experiment ID of the trial                                               |
  | --trial_id, -T  | False    |         | ID of the trial to be found the log path, required when id is not empty. |

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