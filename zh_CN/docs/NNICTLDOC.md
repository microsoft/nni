nnictl

===

## 介绍

**nnictl** is a command line tool, which can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc.

## Commands

nnictl support commands:

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

### Manage an experiment

* **nnictl create**
  
  * Description
    
        You can use this command to create a new experiment, using the configuration specified in config file.
            After this command is successfully done, the context will be set as this experiment,
            which means the following command you issued is associated with this experiment,
            unless you explicitly changes the context(not supported yet).
        
  
  * Usage
    
    ```bash
    nnictl create [OPTIONS]
    ```
    
    Options:  
    
    
    | Name, shorthand | Required | Default | Description                           |
    | --------------- | -------- | ------- | ------------------------------------- |
    | --config, -c    | True     |         | yaml configure file of the experiment |
    | --port, -p      | False    |         | the port of restful server            |

* **nnictl resume**
  
  * Description
    
    You can use this command to resume a stopped experiment.
  
  * Usage
    
    ```bash
    nnictl resume [OPTIONS]
    ```
    
    Options:
    
    | Name, shorthand | Required | Default | Description                                    |
    | --------------- | -------- | ------- | ---------------------------------------------- |
    | id              | False    |         | The id of the experiment you want to resume    |
    | --port, -p      | False    |         | Rest port of the experiment you want to resume |

* **nnictl stop**
  
  * Description
    
    You can use this command to stop a running experiment or multiple experiments.
  
  * Usage
    
    ```bash
    nnictl stop [id]
    ```
  
  * Detail
    
    1. If there is an id specified, and the id matches the running experiment, nnictl will stop the corresponding experiment, or will print error message.
    
    2. If there is no id specified, and there is an experiment running, stop the running experiment, or print error message.
    
    3. If the id ends with *, nnictl will stop all experiments whose ids matchs the regular.
    
    4. If the id does not exist but match the prefix of an experiment id, nnictl will stop the matched experiment.
    
    5. If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information.
    
    6. Users could use 'nnictl stop all' to stop all experiments

* **nnictl update**
  
  * **nnictl update searchspace**
    
    * Description
      
      You can use this command to update an experiment's search space.
    
    * Usage
      
      ```bash
      nnictl update searchspace [OPTIONS]
      ```
      
      Options:
      
      | Name, shorthand | Required | Default | Description                            |
      | --------------- | -------- | ------- | -------------------------------------- |
      | id              | False    |         | ID of the experiment you want to set   |
      | --filename, -f  | True     |         | the file storing your new search space |
  
  * **nnictl update concurrency**
    
    * Description
      
      You can use this command to update an experiment's concurrency.
    
    * Usage
      
      ```bash
      nnictl update concurrency [OPTIONS]
      ```
      
      Options:
      
      | Name, shorthand | Required | Default | Description                             |
      | --------------- | -------- | ------- | --------------------------------------- |
      | id              | False    |         | ID of the experiment you want to set    |
      | --value, -v     | True     |         | the number of allowed concurrent trials |
    
    * **nnictl update duration**
      
      * Description
        
        You can use this command to update an experiment's concurrency.
      
      * Usage
        
        ```bash
        nnictl update duration [OPTIONS]
        ```
        
        Options:
        
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
        
        Options:
        
        | Name, shorthand | Required | Default | Description                                   |
        | --------------- | -------- | ------- | --------------------------------------------- |
        | id              | False    |         | ID of the experiment you want to set          |
        | --value, -v     | True     |         | the new number of maxtrialnum you want to set |

* **nnictl trial**
  
  * **nnictl trial ls**
    
    * Description
      
      You can use this command to show trial's information.
    
    * Usage
      
      ```bash
      nnictl trial ls
      ```
      
      Options:
      
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
      
      Options:
      
      | Name, shorthand | Required | Default | Description                          |
      | --------------- | -------- | ------- | ------------------------------------ |
      | id              | False    |         | ID of the experiment you want to set |
      | --trialid, -t   | True     |         | ID of the trial you want to kill.    |
  
  * **nnictl top**
    
    * Description
      
      Monitor all of running experiments.
    
    * Usage
      
      ```bash
      nnictl top
      ```
      
      Options:
      
      | Name, shorthand | Required | Default | Description                                                                                                  |
      | --------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------ |
      | id              | False    |         | ID of the experiment you want to set                                                                         |
      | --time, -t      | False    |         | The interval to update the experiment status, the unit of time is second, and the default value is 3 second. |

### Manage experiment information

* **nnictl experiment show**
  
  * Description
    
    Show the information of experiment.
  
  * Usage
    
    ```bash
    nnictl experiment show
    ```
    
    Options:
    
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
    
    Options:
    
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