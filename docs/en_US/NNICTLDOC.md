# nnictl

## Introduction

__nnictl__ is a command line tool, which can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc.

## Commands

nnictl support commands:

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

### Manage an experiment

<a name="create"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl create`

* Description

  You can use this command to create a new experiment, using the configuration specified in config file. 

  After this command is successfully done, the context will be set as this experiment, which means the following command you issued is associated with this experiment, unless you explicitly changes the context(not supported yet).

* Usage

  ```bash
  nnictl create [OPTIONS]
  ```

* Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------|------|
  |--config, -c|  True| |YAML configure file of the experiment|
  |--port, -p|False| |the port of restful server|
  |--debug, -d|False||set debug mode|

* Examples

  > create a new experiment with the default port: 8080

  ```bash
  nnictl create --config nni/examples/trials/mnist/config.yml
  ```

  > create a new experiment with specified port 8088

  ```bash
  nnictl create --config nni/examples/trials/mnist/config.yml --port 8088
  ```

  > create a new experiment with specified port 8088 and debug mode

  ```bash
  nnictl create --config nni/examples/trials/mnist/config.yml --port 8088 --debug
  ```

Note:

```text
Debug mode will disable version check function in Trialkeeper.
```

<a name="resume"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl resume`

* Description

  You can use this command to resume a stopped experiment.

* Usage

  ```bash
  nnictl resume [OPTIONS]
  ```

* Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  True| |The id of the experiment you want to resume|  
  |--port, -p|  False| |Rest port of the experiment you want to resume|
  |--debug, -d|False||set debug mode|

* Example

  > resume an experiment with specified port 8088

  ```bash
  nnictl resume [experiment_id] --port 8088
  ```

<a name="stop"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl stop`

* Description

  You can use this command to stop a running experiment or multiple experiments.

* Usage

  ```bash
  nnictl stop [id]
  ```

* Details & Examples

  1. If there is no id specified, and there is an experiment running, stop the running experiment, or print error message.

      ```bash
      nnictl stop
      ```

  1. If there is an id specified, and the id matches the running experiment, nnictl will stop the corresponding experiment, or will print error message.

      ```bash
      nnictl stop [experiment_id]
      ```

  1. Users could use 'nnictl stop all' to stop all experiments.

      ```bash
      nnictl stop all
      ```

  1. If the id ends with *, nnictl will stop all experiments whose ids matchs the regular.
  1. If the id does not exist but match the prefix of an experiment id, nnictl will stop the matched experiment.
  1. If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information.

<a name="update"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `nnictl update`

* __nnictl update searchspace__
  * Description

    You can use this command to update an experiment's search space.

  * Usage

    ```bash
    nnictl update searchspace [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--filename, -f|  True| |the file storing your new search space|

  * Example

    `update experiment's new search space with file dir 'examples/trials/mnist/search_space.json'`

    ```bash
    nnictl update searchspace [experiment_id] --filename examples/trials/mnist/search_space.json
    ```

* __nnictl update concurrency__  

  * Description

     You can use this command to update an experiment's concurrency.

  * Usage

    ```bash
    nnictl update concurrency [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--value, -v|  True| |the number of allowed concurrent trials|

  * Example

    > update experiment's concurrency

    ```bash
    nnictl update concurrency [experiment_id] --value [concurrency_number]
    ```

* __nnictl update duration__  

  * Description

    You can use this command to update an experiment's duration.  

  * Usage

    ```bash
    nnictl update duration [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--value, -v|  True| |the experiment duration will be NUMBER seconds. SUFFIX may be 's' for seconds (the default), 'm' for minutes, 'h' for hours or 'd' for days.|

  * Example

    > update experiment's duration

    ```bash
    nnictl update duration [experiment_id] --value [duration]
    ```

* __nnictl update trialnum__  
  * Description

    You can use this command to update an experiment's maxtrialnum.

  * Usage

    ```bash
    nnictl update trialnum [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--value, -v|  True| |the new number of maxtrialnum you want to set|

  * Example

    > update experiment's trial num

    ```bash
    nnictl update trialnum --id [experiment_id] --value [trial_num]
    ```

<a name="trial"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `nnictl trial`

* __nnictl trial ls__

  * Description

    You can use this command to show trial's information.

  * Usage

    ```bash
    nnictl trial ls
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|

* __nnictl trial kill__

  * Description

    You can use this command to kill a trial job.

  * Usage

    ```bash
    nnictl trial kill [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |Experiment ID of the trial|
  |--trial_id, -T|  True| |ID of the trial you want to kill.|

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

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--time, -t|  False| |The interval to update the experiment status, the unit of time is second, and the default value is 3 second.|

<a name="experiment"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `Manage experiment information`

* __nnictl experiment show__

  * Description

    Show the information of experiment.

  * Usage

    ```bash
    nnictl experiment show
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|

* __nnictl experiment status__

  * Description

    Show the status of experiment.

  * Usage

    ```bash
    nnictl experiment status
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|

* __nnictl experiment list__

  * Description

    Show the information of all the (running) experiments.

  * Usage

    ```bash
    nnictl experiment list
    ```

<a name="export"></a>

* __nnictl experiment export__
  * Description

    You can use this command to export reward & hyper-parameter of trial jobs to a csv file.

  * Usage

    ```bash
    nnictl experiment export [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment    |
  |--filename, -f|  True| |File path of the output file     |
  |--type|  True| |Type of output file, only support "csv" and "json"|

  * Examples

  > export all trial data in an experiment as json format

  ```bash
  nnictl experiment export [experiment_id] --filename [file_path] --type json
  ```

* __nnictl experiment import__
  * Description

    You can use this command to import several prior or supplementary trial hyperparameters & results for NNI hyperparameter tuning. The data are fed to the tuning algorithm (e.g., tuner or advisor).

  * Usage

    ```bash
    nnictl experiment import [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------|------|
  |id|  False| |The id of the experiment you want to import data into|
  |--filename, -f|  True| |a file with data you want to import in json format|

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

* __nnictl log stdout__

  * Description

    Show the stdout log content.

  * Usage

    ```bash
    nnictl log stdout [options]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--head, -h| False| |show head lines of stdout|
  |--tail, -t|  False| |show tail lines of stdout|
  |--path, -p|  False| |show the path of stdout file|

  * Example

    > Show the tail of stdout log content

    ```bash
    nnictl log stdout [experiment_id] --tail [lines_number]
    ```

* __nnictl log stderr__
  * Description

    Show the stderr log content.

  * Usage

    ```bash
    nnictl log stderr [options]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--head, -h| False| |show head lines of stderr|
  |--tail, -t|  False| |show tail lines of stderr|
  |--path, -p|  False| |show the path of stderr file|

* __nnictl log trial__

  * Description
  
    Show trial log path.
  
  * Usage

    ```bash  
    nnictl log trial [options]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |Experiment ID of the trial|
  |--trial_id, -T|  False| |ID of the trial to be found the log path, required when id is not empty.|

<a name="webui"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `Manage webui`

* __nnictl webui url__

<a name="tensorboard"></a>
![](https://placehold.it/15/1589F0/000000?text=+) `Manage tensorboard`

* __nnictl tensorboard start__

  * Description

    Start the tensorboard process.
  
  * Usage

    ```bash
    nnictl tensorboard start
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|
  |--trial_id, -T|  False| |ID of the trial|
  |--port|  False| 6006|The port of the tensorboard process|

  * Detail

    1. NNICTL support tensorboard function in local and remote platform for the moment, other platforms will be supported later.   
    2. If you want to use tensorboard, you need to write your tensorboard log data to environment variable [NNI_OUTPUT_DIR] path.  
    3. In local mode, nnictl will set --logdir=[NNI_OUTPUT_DIR] directly and start a tensorboard process.
    4. In remote mode, nnictl will create a ssh client to copy log data from remote machine to local temp directory firstly, and then start a tensorboard process in your local machine. You need to notice that nnictl only copy the log data one time when you use the command, if you want to see the later result of tensorboard, you should execute nnictl tensorboard command again.
    5. If there is only one trial job, you don't need to set trial id. If there are multiple trial jobs running, you should set the trial id, or you could use [nnictl tensorboard start --trial_id all] to map --logdir to all trial log paths.

* __nnictl tensorboard stop__
  * Description

    Stop all of the tensorboard process.

  * Usage

    ```bash
    nnictl tensorboard stop
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |id|  False| |ID of the experiment you want to set|

<a name="package"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Manage package`

* __nnictl package install__
  * Description

    Install the packages needed in nni experiments.

  * Usage

    ```bash
    nnictl package install [OPTIONS]
    ```

  * Options

  |Name, shorthand|Required|Default|Description|
  |------|------|------ |------|
  |--name|  True| |The name of package to be installed|

  * Example

    > Install the packages needed in tuner SMAC

    ```bash
    nnictl package install --name=SMAC
    ```

* __nnictl package show__

  * Description

    List the packages supported.

  * Usage

    ```bash
    nnictl package show
    ```

<a name="version"></a>

![](https://placehold.it/15/1589F0/000000?text=+) `Check NNI version`

* __nnictl --version__

  * Description

    Describe the current version of NNI installed.

  * Usage

    ```bash
    nnictl --version
    ```
