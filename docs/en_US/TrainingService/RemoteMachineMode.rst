Run an Experiment on Remote Machines
====================================

NNI can run one experiment on multiple remote machines through SSH, called ``remote`` mode. It's like a lightweight training platform. In this mode, NNI can be started from your computer, and dispatch trials to remote machines in parallel.

The OS of remote machines supports ``Linux``\ , ``Windows 10``\ , and ``Windows Server 2019``.

Requirements
------------


* 
  Make sure the default environment of remote machines meets requirements of your trial code. If the default environment does not meet the requirements, the setup script can be added into ``command`` field of NNI config.

* 
  Make sure remote machines can be accessed through SSH from the machine which runs ``nnictl`` command. It supports both password and key authentication of SSH. For advanced usages, please refer to `machineList part of configuration <../Tutorial/ExperimentConfig.rst>`__.

* 
  Make sure the NNI version on each machine is consistent.

* 
  Make sure the command of Trial is compatible with remote OSes, if you want to use remote Linux and Windows together. For example, the default python 3.x executable called ``python3`` on Linux, and ``python`` on Windows.

Linux
^^^^^


* Follow `installation <../Tutorial/InstallationLinux.rst>`__ to install NNI on the remote machine.

Windows
^^^^^^^


* 
  Follow `installation <../Tutorial/InstallationWin.rst>`__ to install NNI on the remote machine.

* 
  Install and start ``OpenSSH Server``.


  #. 
     Open ``Settings`` app on Windows.

  #. 
     Click ``Apps``\ , then click ``Optional features``.

  #. 
     Click ``Add a feature``\ , search and select ``OpenSSH Server``\ , and then click ``Install``.

  #. 
     Once it's installed, run below command to start and set to automatic start.

  .. code-block:: bat

     sc config sshd start=auto
     net start sshd

* 
  Make sure remote account is administrator, so that it can stop running trials.

* 
  Make sure there is no welcome message more than default, since it causes ssh2 failed in NodeJs. For example, if you're using Data Science VM on Azure, it needs to remove extra echo commands in ``C:\dsvm\tools\setup\welcome.bat``.

  The output like below is ok, when opening a new command window.

  .. code-block:: text

     Microsoft Windows [Version 10.0.17763.1192]
     (c) 2018 Microsoft Corporation. All rights reserved.

     (py37_default) C:\Users\AzureUser>

Run an experiment
-----------------

e.g. there are three machines, which can be logged in with username and password.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - IP
     - Username
     - Password
   * - 10.1.1.1
     - bob
     - bob123
   * - 10.1.1.2
     - bob
     - bob123
   * - 10.1.1.3
     - bob
     - bob123


Install and run NNI on one of those three machines or another machine, which has network access to them.

Use ``examples/trials/mnist-annotation`` as the example. Below is content of ``examples/trials/mnist-annotation/config_remote.yml``\ :

.. code-block:: yaml

   authorName: default
   experimentName: example_mnist
   trialConcurrency: 1
   maxExecDuration: 1h
   maxTrialNum: 10
   #choice: local, remote, pai
   trainingServicePlatform: remote
   # search space file
   searchSpacePath: search_space.json
   #choice: true, false
   useAnnotation: true
   tuner:
     #choice: TPE, Random, Anneal, Evolution, BatchTuner
     #SMAC (SMAC should be installed through nnictl)
     builtinTunerName: TPE
     classArgs:
       #choice: maximize, minimize
       optimize_mode: maximize
   trial:
     command: python3 mnist.py
     codeDir: .
     gpuNum: 0
   #machineList can be empty if the platform is local
   machineList:
     - ip: 10.1.1.1
       username: bob
       passwd: bob123
       #port can be skip if using default ssh port 22
       #port: 22
     - ip: 10.1.1.2
       username: bob
       passwd: bob123
     - ip: 10.1.1.3
       username: bob
       passwd: bob123

Files in ``codeDir`` will be uploaded to remote machines automatically. You can run below command on Windows, Linux, or macOS to spawn trials on remote Linux machines:

.. code-block:: bash

   nnictl create --config examples/trials/mnist-annotation/config_remote.yml

Configure python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, commands and scripts will be executed in the default environment in remote machine. If there are multiple python virtual environments in your remote machine, and you want to run experiments in a specific environment, then use **preCommand** to specify a python environment on your remote machine. 

Use ``examples/trials/mnist-tfv2`` as the example. Below is content of ``examples/trials/mnist-tfv2/config_remote.yml``\ :

.. code-block:: yaml

   authorName: default
   experimentName: example_mnist
   trialConcurrency: 1
   maxExecDuration: 1h
   maxTrialNum: 10
   #choice: local, remote, pai
   trainingServicePlatform: remote
   searchSpacePath: search_space.json
   #choice: true, false
   useAnnotation: false
   tuner:
     #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
     #SMAC (SMAC should be installed through nnictl)
     builtinTunerName: TPE
     classArgs:
       #choice: maximize, minimize
       optimize_mode: maximize
   trial:
     command: python3 mnist.py
     codeDir: .
     gpuNum: 0
   #machineList can be empty if the platform is local
   machineList:
     - ip: ${replace_to_your_remote_machine_ip}
       username: ${replace_to_your_remote_machine_username}
       sshKeyPath: ${replace_to_your_remote_machine_sshKeyPath}
       # Pre-command will be executed before the remote machine executes other commands.
       # Below is an example of specifying python environment.
       # If you want to execute multiple commands, please use "&&" to connect them.
       # preCommand: source ${replace_to_absolute_path_recommended_here}/bin/activate
       # preCommand: source ${replace_to_conda_path}/bin/activate ${replace_to_conda_env_name}
       preCommand: export PATH=${replace_to_python_environment_path_in_your_remote_machine}:$PATH

The **preCommand** will be executed before the remote machine executes other commands. So you can configure python environment path like this:

.. code-block:: yaml

   # Linux remote machine
   preCommand: export PATH=${replace_to_python_environment_path_in_your_remote_machine}:$PATH
   # Windows remote machine
   preCommand: set path=${replace_to_python_environment_path_in_your_remote_machine};%path%

Or if you want to activate the ``virtualenv`` environment:

.. code-block:: yaml

   # Linux remote machine
   preCommand: source ${replace_to_absolute_path_recommended_here}/bin/activate
   # Windows remote machine
   preCommand: ${replace_to_absolute_path_recommended_here}\\scripts\\activate

Or if you want to activate the ``conda`` environment:

.. code-block:: yaml

   # Linux remote machine
   preCommand: source ${replace_to_conda_path}/bin/activate ${replace_to_conda_env_name}
   # Windows remote machine
   preCommand: call activate ${replace_to_conda_env_name}

If you want multiple commands to be executed, you can use ``&&`` to connect these commands:

.. code-block:: yaml

   preCommand: command1 && command2 && command3

**Note**\ : Because **preCommand** will execute before other commands each time, it is strongly not recommended to set **preCommand** that will make changes to system, i.e. ``mkdir`` or ``touch``.
