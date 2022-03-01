Remote Training Service
=======================

NNI can run one experiment on multiple remote machines through SSH, called ``remote`` mode. It's like a lightweight training platform. In this mode, NNI can be started from your computer, and dispatch trials to remote machines in parallel.

The OS of remote machines supports ``Linux``\ , ``Windows 10``\ , and ``Windows Server 2019``.

Prerequisite
------------


1. Make sure the default environment of remote machines meets requirements of your trial code. If the default environment does not meet the requirements, the setup script can be added into ``command`` field of NNI config.

2. Make sure remote machines can be accessed through SSH from the machine which runs ``nnictl`` command. It supports both password and key authentication of SSH. For advanced usage, please refer to :ref:`reference-remote-config-label` in reference for detailed usage.

3. Make sure the NNI version on each machine is consistent. Follow the install guide `here <../Tutorial/QuickStart.rst>`__ to install NNI.

4. Make sure the command of Trial is compatible with remote OSes, if you want to use remote Linux and Windows together. For example, the default python 3.x executable called ``python3`` on Linux, and ``python`` on Windows.

There are several steps for Windows server.

1. Install and start ``OpenSSH Server``.

    1) Open ``Settings`` app on Windows.

    2) Click ``Apps``\ , then click ``Optional features``.

    3) Click ``Add a feature``\ , search and select ``OpenSSH Server``\ , and then click ``Install``.

    4) Once it's installed, run below command to start and set to automatic start.

    .. code-block:: bat

        sc config sshd start=auto
        net start sshd

2. Make sure remote account is administrator, so that it can stop running trials.

3. Make sure there is no welcome message more than default, since it causes ssh2 failed in NodeJs. For example, if you're using Data Science VM on Azure, it needs to remove extra echo commands in ``C:\dsvm\tools\setup\welcome.bat``.

  The output like below is ok, when opening a new command window.

  .. code-block:: text

     Microsoft Windows [Version 10.0.17763.1192]
     (c) 2018 Microsoft Corporation. All rights reserved.

     (py37_default) C:\Users\AzureUser>

Usage
-----

Use ``examples/trials/mnist-pytorch`` as the example. Suppose there are two machines, which can be logged in with username and password or key authentication of SSH. Install and run NNI on one of those machines or another machine, which has network access to them. Here is a template configuration specification.

.. code-block:: yaml

   searchSpaceFile: search_space.json
   trialCommand: python3 mnist.py
   trialCodeDirectory: .  # default value, can be omitted
   trialGpuNumber: 0
   trialConcurrency: 4
   maxTrialNumber: 20
   tuner:
     name: TPE
     classArgs:
       optimize_mode: maximize
   trainingService:
     platform: remote
     machineList:
       - host: 192.0.2.1
         user: alice
         ssh_key_file: ~/.ssh/id_rsa
       - host: 192.0.2.2
         port: 10022
         user: bob
         password: bob123

The example configuration is saved in ``examples/trials/mnist-pytorch/config_remote.yml``.

Files in ``trialCodeDirectory`` will be uploaded to remote machines automatically. You can run below command on Windows, Linux, or macOS to spawn trials on remote Linux machines:

.. code-block:: bash

   nnictl create --config examples/trials/mnist-pytorch/config_remote.yml

More features
-------------

Configure python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, commands and scripts will be executed in the default environment in remote machine. If there are multiple python virtual environments in your remote machine, and you want to run experiments in a specific environment, then use **pythonPath** to specify a python environment on your remote machine. 

For example, with anaconda you can specify:

.. code-block:: yaml

   pythonPath: /home/bob/.conda/envs/ENV-NAME/bin

Configure distributed trial
^^^^^^^^^^^^^^^^^^^^^^^^^^^

(some training service, e.g., openpai, kubeflow, already supported distributed trial)


Monitor via TensorBoard
^^^^^^^^^^^^^^^^^^^^^^^
