.. role:: raw-html(raw)
   :format: html


nnictl
======

Introduction
------------

**nnictl** is a command line tool, which can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc.

Commands
--------

nnictl support commands:


* `nnictl create <#create>`__
* `nnictl resume <#resume>`__
* `nnictl view <#view>`__
* `nnictl stop <#stop>`__
* `nnictl update <#update>`__
* `nnictl trial <#trial>`__
* `nnictl top <#top>`__
* `nnictl experiment <#experiment>`__
* `nnictl platform <#platform>`__
* `nnictl config <#config>`__
* `nnictl log <#log>`__
* `nnictl webui <#webui>`__
* `nnictl tensorboard <#tensorboard>`__
* `nnictl package <#package>`__
* `nnictl ss_gen <#ss_gen>`__
* `nnictl --version <#version>`__

Manage an experiment
^^^^^^^^^^^^^^^^^^^^

:raw-html:`<a name="create"></a>`

nnictl create
^^^^^^^^^^^^^


* 
  Description

  You can use this command to create a new experiment, using the configuration specified in config file.

  After this command is successfully done, the context will be set as this experiment, which means the following command you issued is associated with this experiment, unless you explicitly changes the context(not supported yet).

* 
  Usage

  .. code-block:: bash

     nnictl create [OPTIONS]

* 
  Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - --config, -c
     - True
     - 
     - YAML configure file of the experiment
   * - --port, -p
     - False
     - 
     - the port of restful server
   * - --debug, -d
     - False
     - 
     - set debug mode
   * - --foreground, -f
     - False
     - 
     - set foreground mode, print log content to terminal



* 
  Examples

  ..

     create a new experiment with the default port: 8080


  .. code-block:: bash

     nnictl create --config nni/examples/trials/mnist-tfv1/config.yml

  ..

     create a new experiment with specified port 8088


  .. code-block:: bash

     nnictl create --config nni/examples/trials/mnist-tfv1/config.yml --port 8088

  ..

     create a new experiment with specified port 8088 and debug mode


  .. code-block:: bash

     nnictl create --config nni/examples/trials/mnist-tfv1/config.yml --port 8088 --debug

Note:

.. code-block:: text

   Debug mode will disable version check function in Trialkeeper.

:raw-html:`<a name="resume"></a>`

nnictl resume
^^^^^^^^^^^^^


* 
  Description

  You can use this command to resume a stopped experiment.

* 
  Usage

  .. code-block:: bash

     nnictl resume [OPTIONS]

* 
  Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - True
     - 
     - The id of the experiment you want to resume
   * - --port, -p
     - False
     - 
     - Rest port of the experiment you want to resume
   * - --debug, -d
     - False
     - 
     - set debug mode
   * - --foreground, -f
     - False
     - 
     - set foreground mode, print log content to terminal



* 
  Example

  ..

     resume an experiment with specified port 8088


  .. code-block:: bash

     nnictl resume [experiment_id] --port 8088

:raw-html:`<a name="view"></a>`

nnictl view
^^^^^^^^^^^


* 
  Description

  You can use this command to view a stopped experiment.

* 
  Usage

  .. code-block:: bash

     nnictl view [OPTIONS]

* 
  Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - True
     - 
     - The id of the experiment you want to view
   * - --port, -p
     - False
     - 
     - Rest port of the experiment you want to view



* 
  Example

  ..

     view an experiment with specified port 8088


  .. code-block:: bash

     nnictl view [experiment_id] --port 8088

:raw-html:`<a name="stop"></a>`

nnictl stop
^^^^^^^^^^^


* 
  Description

  You can use this command to stop a running experiment or multiple experiments.

* 
  Usage

  .. code-block:: bash

     nnictl stop [Options]

* 
  Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - The id of the experiment you want to stop
   * - --port, -p
     - False
     - 
     - Rest port of the experiment you want to stop
   * - --all, -a
     - False
     - 
     - Stop all of experiments



* 
  Details & Examples


  #. 
     If there is no id specified, and there is an experiment running, stop the running experiment, or print error message.

     .. code-block:: bash

         nnictl stop

  #. 
     If there is an id specified, and the id matches the running experiment, nnictl will stop the corresponding experiment, or will print error message.

     .. code-block:: bash

         nnictl stop [experiment_id]

  #. 
     If there is a port specified, and an experiment is running on that port, the experiment will be stopped.

     .. code-block:: bash

         nnictl stop --port 8080

  #. 
     Users could use 'nnictl stop --all' to stop all experiments.

     .. code-block:: bash

         nnictl stop --all

  #. 
     If the id ends with \*, nnictl will stop all experiments whose ids matchs the regular.

  #. If the id does not exist but match the prefix of an experiment id, nnictl will stop the matched experiment.
  #. If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information.

:raw-html:`<a name="update"></a>`

nnictl update
^^^^^^^^^^^^^


* 
  **nnictl update searchspace**


  * 
    Description

    You can use this command to update an experiment's search space.

  * 
    Usage

    .. code-block:: bash

       nnictl update searchspace [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --filename, -f
     - True
     - 
     - the file storing your new search space



* 
  Example

  ``update experiment's new search space with file dir 'examples/trials/mnist-tfv1/search_space.json'``

  .. code-block:: bash

     nnictl update searchspace [experiment_id] --filename examples/trials/mnist-tfv1/search_space.json


* 
  **nnictl update concurrency**


  * 
    Description

     You can use this command to update an experiment's concurrency.

  * 
    Usage

    .. code-block:: bash

       nnictl update concurrency [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --value, -v
     - True
     - 
     - the number of allowed concurrent trials



* 
  Example

  ..

     update experiment's concurrency


  .. code-block:: bash

     nnictl update concurrency [experiment_id] --value [concurrency_number]


* 
  **nnictl update duration**


  * 
    Description

    You can use this command to update an experiment's duration.

  * 
    Usage

    .. code-block:: bash

       nnictl update duration [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --value, -v
     - True
     - 
     - Strings like '1m' for one minute or '2h' for two hours. SUFFIX may be 's' for seconds, 'm' for minutes, 'h' for hours or 'd' for days.



* 
  Example

  ..

     update experiment's duration


  .. code-block:: bash

     nnictl update duration [experiment_id] --value [duration]


* 
  **nnictl update trialnum**


  * 
    Description

    You can use this command to update an experiment's maxtrialnum.

  * 
    Usage

    .. code-block:: bash

       nnictl update trialnum [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --value, -v
     - True
     - 
     - the new number of maxtrialnum you want to set



* 
  Example

  ..

     update experiment's trial num


  .. code-block:: bash

     nnictl update trialnum [experiment_id] --value [trial_num]

:raw-html:`<a name="trial"></a>`

nnictl trial
^^^^^^^^^^^^


* 
  **nnictl trial ls**


  * 
    Description

    You can use this command to show trial's information. Note that if ``head`` or ``tail`` is set, only complete trials will be listed.

  * 
    Usage

    .. code-block:: bash

       nnictl trial ls
       nnictl trial ls --head 10
       nnictl trial ls --tail 10

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --head
     - False
     - 
     - the number of items to be listed with the highest default metric
   * - --tail
     - False
     - 
     - the number of items to be listed with the lowest default metric



* 
  **nnictl trial kill**


  * 
    Description

    You can use this command to kill a trial job.

  * 
    Usage

    .. code-block:: bash

       nnictl trial kill [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - Experiment ID of the trial
   * - --trial_id, -T
     - True
     - 
     - ID of the trial you want to kill.



* 
  Example

  ..

     kill trail job


  .. code-block:: bash

     nnictl trial kill [experiment_id] --trial_id [trial_id]

:raw-html:`<a name="top"></a>`

nnictl top
^^^^^^^^^^


* 
  Description

  Monitor all of running experiments.

* 
  Usage

  .. code-block:: bash

     nnictl top

* 
  Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --time, -t
     - False
     - 
     - The interval to update the experiment status, the unit of time is second, and the default value is 3 second.


:raw-html:`<a name="experiment"></a>`

Manage experiment information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  **nnictl experiment show**


  * 
    Description

    Show the information of experiment.

  * 
    Usage

    .. code-block:: bash

       nnictl experiment show

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set



* 
  **nnictl experiment status**


  * 
    Description

    Show the status of experiment.

  * 
    Usage

    .. code-block:: bash

       nnictl experiment status

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set



* 
  **nnictl experiment list**


  * 
    Description

    Show the information of all the (running) experiments.

  * 
    Usage

    .. code-block:: bash

       nnictl experiment list [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - --all
     - False
     - 
     - list all of experiments



* 
  **nnictl experiment delete**


  * 
    Description

    Delete one or all experiments, it includes log, result, environment information and cache. It uses to delete useless experiment result, or save disk space.

  * 
    Usage

    .. code-block:: bash

       nnictl experiment delete [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment
   * - --all
     - False
     - 
     - delete all of experiments



* 
  **nnictl experiment export**


  * 
    Description

    You can use this command to export reward & hyper-parameter of trial jobs to a csv file.

  * 
    Usage

    .. code-block:: bash

       nnictl experiment export [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment
   * - --filename, -f
     - True
     - 
     - File path of the output file
   * - --type
     - True
     - 
     - Type of output file, only support "csv" and "json"
   * - --intermediate, -i
     - False
     - 
     - Are intermediate results included



* 
  Examples

  ..

     export all trial data in an experiment as json format


  .. code-block:: bash

     nnictl experiment export [experiment_id] --filename [file_path] --type json --intermediate


* 
  **nnictl experiment import**


  * 
    Description

    You can use this command to import several prior or supplementary trial hyperparameters & results for NNI hyperparameter tuning. The data are fed to the tuning algorithm (e.g., tuner or advisor).

  * 
    Usage

    .. code-block:: bash

       nnictl experiment import [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - The id of the experiment you want to import data into
   * - --filename, -f
     - True
     - 
     - a file with data you want to import in json format



* 
  Details

  NNI supports users to import their own data, please express the data in the correct format. An example is shown below:

  .. code-block:: json

     [
       {"parameter": {"x": 0.5, "y": 0.9}, "value": 0.03},
       {"parameter": {"x": 0.4, "y": 0.8}, "value": 0.05},
       {"parameter": {"x": 0.3, "y": 0.7}, "value": 0.04}
     ]

  Every element in the top level list is a sample. For our built-in tuners/advisors, each sample should have at least two keys: ``parameter`` and ``value``. The ``parameter`` must match this experiment's search space, that is, all the keys (or hyperparameters) in ``parameter`` must match the keys in the search space. Otherwise, tuner/advisor may have unpredictable behavior. ``Value`` should follow the same rule of the input in ``nni.report_final_result``\ , that is, either a number or a dict with a key named ``default``. For your customized tuner/advisor, the file could have any json content depending on how you implement the corresponding methods (e.g., ``import_data``\ ).

  You also can use `nnictl experiment export <#export>`__ to export a valid json file including previous experiment trial hyperparameters and results.

  Currently, following tuner and advisor support import data:

  .. code-block:: yaml

     builtinTunerName: TPE, Anneal, GridSearch, MetisTuner
     builtinAdvisorName: BOHB

  *If you want to import data to BOHB advisor, user are suggested to add "TRIAL_BUDGET" in parameter as NNI do, otherwise, BOHB will use max_budget as "TRIAL_BUDGET". Here is an example:*

  .. code-block:: json

     [
       {"parameter": {"x": 0.5, "y": 0.9, "TRIAL_BUDGET": 27}, "value": 0.03}
     ]

* 
  Examples

  ..

     import data to a running experiment


  .. code-block:: bash

     nnictl experiment import [experiment_id] -f experiment_data.json


* 
  **nnictl experiment save**


  * 
    Description

    Save nni experiment metadata and code data.

  * 
    Usage

    .. code-block:: bash

       nnictl experiment save [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - True
     - 
     - The id of the experiment you want to save
   * - --path, -p
     - False
     - 
     - the folder path to store nni experiment data, default current working directory
   * - --saveCodeDir, -s
     - False
     - 
     - save codeDir data of the experiment, default False



* 
  Examples

  ..

     save an expeirment


  .. code-block:: bash

     nnictl experiment save [experiment_id] --saveCodeDir


* 
  **nnictl experiment load**


  * 
    Description

    Load an nni experiment.

  * 
    Usage

    .. code-block:: bash

       nnictl experiment load [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - --path, -p
     - True
     - 
     - the file path of nni package
   * - --codeDir, -c
     - True
     - 
     - the path of codeDir for loaded experiment, this path will also put the code in the loaded experiment package
   * - --logDir, -l
     - False
     - 
     - the path of logDir for loaded experiment
   * - --searchSpacePath, -s
     - True
     - 
     - the path of search space file for loaded experiment, this path contains file name. Default in $codeDir/search_space.json



* 
  Examples

  ..

     load an expeirment


  .. code-block:: bash

     nnictl experiment load --path [path] --codeDir [codeDir]

:raw-html:`<a name="platform"></a>`

Manage platform information
^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  **nnictl platform clean**


  * 
    Description

    It uses to clean up disk on a target platform. The provided YAML file includes the information of target platform, and it follows the same schema as the NNI configuration file.

  * 
    Note

    if the target platform is being used by other users, it may cause unexpected errors to others.

  * 
    Usage

    .. code-block:: bash

       nnictl platform clean [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - --config
     - True
     - 
     - the path of yaml config file used when create an experiment


:raw-html:`<a name="config"></a>`

nnictl config show
^^^^^^^^^^^^^^^^^^


* 
  Description

  Display the current context information.

* 
  Usage

  .. code-block:: bash

     nnictl config show

:raw-html:`<a name="log"></a>`

Manage log
^^^^^^^^^^


* 
  **nnictl log stdout**


  * 
    Description

    Show the stdout log content.

  * 
    Usage

    .. code-block:: bash

       nnictl log stdout [options]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --head, -h
     - False
     - 
     - show head lines of stdout
   * - --tail, -t
     - False
     - 
     - show tail lines of stdout
   * - --path, -p
     - False
     - 
     - show the path of stdout file



* 
  Example

  ..

     Show the tail of stdout log content


  .. code-block:: bash

     nnictl log stdout [experiment_id] --tail [lines_number]


* 
  **nnictl log stderr**


  * 
    Description

    Show the stderr log content.

  * 
    Usage

    .. code-block:: bash

       nnictl log stderr [options]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --head, -h
     - False
     - 
     - show head lines of stderr
   * - --tail, -t
     - False
     - 
     - show tail lines of stderr
   * - --path, -p
     - False
     - 
     - show the path of stderr file



* 
  **nnictl log trial**


  * 
    Description

    Show trial log path.

  * 
    Usage

    .. code-block:: bash

       nnictl log trial [options]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - Experiment ID of the trial
   * - --trial_id, -T
     - False
     - 
     - ID of the trial to be found the log path, required when id is not empty.


:raw-html:`<a name="webui"></a>`

Manage webui
^^^^^^^^^^^^


* 
  **nnictl webui url**


  * 
    Description

    Show an experiment's webui url

  * 
    Usage

    .. code-block:: bash

       nnictl webui url [options]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - Experiment ID


:raw-html:`<a name="tensorboard"></a>`

Manage tensorboard
^^^^^^^^^^^^^^^^^^


* 
  **nnictl tensorboard start**


  * 
    Description

    Start the tensorboard process.

  * 
    Usage

    .. code-block:: bash

       nnictl tensorboard start

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set
   * - --trial_id, -T
     - False
     - 
     - ID of the trial
   * - --port
     - False
     - 6006
     - The port of the tensorboard process



* 
  Detail


  #. NNICTL support tensorboard function in local and remote platform for the moment, other platforms will be supported later.
  #. If you want to use tensorboard, you need to write your tensorboard log data to environment variable [NNI_OUTPUT_DIR] path.
  #. In local mode, nnictl will set --logdir=[NNI_OUTPUT_DIR] directly and start a tensorboard process.
  #. In remote mode, nnictl will create a ssh client to copy log data from remote machine to local temp directory firstly, and then start a tensorboard process in your local machine. You need to notice that nnictl only copy the log data one time when you use the command, if you want to see the later result of tensorboard, you should execute nnictl tensorboard command again.
  #. If there is only one trial job, you don't need to set trial id. If there are multiple trial jobs running, you should set the trial id, or you could use [nnictl tensorboard start --trial_id all] to map --logdir to all trial log paths.


* 
  **nnictl tensorboard stop**


  * 
    Description

    Stop all of the tensorboard process.

  * 
    Usage

    .. code-block:: bash

       nnictl tensorboard stop

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - id
     - False
     - 
     - ID of the experiment you want to set


:raw-html:`<a name="package"></a>`

Manage package
^^^^^^^^^^^^^^


* 
  **nnictl package install**


  * 
    Description

    Install a package (customized algorithms or nni provided algorithms) as builtin tuner/assessor/advisor.

  * 
    Usage

    .. code-block:: bash

       nnictl package install --name <package name>

    The available ``<package name>`` can be checked via ``nnictl package list`` command.

    or

    .. code-block:: bash

       nnictl package install <installation source>

    Reference `Install customized algorithms <InstallCustomizedAlgos.rst>`__ to prepare the installation source.

  * 
    Example

    ..

       Install SMAC tuner


    .. code-block:: bash

       nnictl package install --name SMAC

    ..

       Install a customized tuner


    .. code-block:: bash

       nnictl package install nni/examples/tuners/customized_tuner/dist/demo_tuner-0.1-py3-none-any.whl


* 
  **nnictl package show**


  * 
    Description

    Show the detailed information of specified packages.

  * 
    Usage

    .. code-block:: bash

       nnictl package show <package name>

  * 
    Example

    .. code-block:: bash

       nnictl package show SMAC

* 
  **nnictl package list**


  * 
    Description

    List the installed/all packages.

  * 
    Usage

    .. code-block:: bash

       nnictl package list [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - --all
     - False
     - 
     - List all packages



* 
  Example

  ..

     List installed packages


  .. code-block:: bash

     nnictl package list

  ..

     List all packages


  .. code-block:: bash

     nnictl package list --all


* 
  **nnictl package uninstall**


  * 
    Description

    Uninstall a package.

  * 
    Usage

    .. code-block:: bash

       nnictl package uninstall <package name>

  * 
    Example
    Uninstall SMAC package

    .. code-block:: bash

       nnictl package uninstall SMAC

:raw-html:`<a name="ss_gen"></a>`

Generate search space
^^^^^^^^^^^^^^^^^^^^^


* 
  **nnictl ss_gen**


  * 
    Description

    Generate search space from user trial code which uses NNI NAS APIs.

  * 
    Usage

    .. code-block:: bash

       nnictl ss_gen [OPTIONS]

  * 
    Options

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name, shorthand
     - Required
     - Default
     - Description
   * - --trial_command
     - True
     - 
     - The command of the trial code
   * - --trial_dir
     - False
     - ./
     - The directory of the trial code
   * - --file
     - False
     - nni_auto_gen_search_space.json
     - The file for storing generated search space



* 
  Example

  ..

     Generate a search space


  .. code-block:: bash

     nnictl ss_gen --trial_command="python3 mnist.py" --trial_dir=./ --file=ss.json

:raw-html:`<a name="version"></a>`

Check NNI version
^^^^^^^^^^^^^^^^^


* 
  **nnictl --version**


  * 
    Description

    Describe the current version of NNI installed.

  * 
    Usage

    .. code-block:: bash

       nnictl --version
