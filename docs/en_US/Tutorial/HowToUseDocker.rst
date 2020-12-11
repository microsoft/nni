**How to Use Docker in NNI**
================================

Overview
--------

`Docker <https://www.docker.com/>`__ is a tool to make it easier for users to deploy and run applications based on their own operating system by starting containers. Docker is not a virtual machine, it does not create a virtual operating system, but it allows different applications to use the same OS kernel and isolate different applications by container.

Users can start NNI experiments using Docker. NNI also provides an official Docker image `msranni/nni <https://hub.docker.com/r/msranni/nni>`__ on Docker Hub.

Using Docker in local machine
-----------------------------

Step 1: Installation of Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you start using Docker for NNI experiments, you should install Docker on your local machine. `See here <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`__.

Step 2: Start a Docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have installed the Docker package in your local machine, you can start a Docker container instance to run NNI examples. You should notice that because NNI will start a web UI process in a container and continue to listen to a port, you need to specify the port mapping between your host machine and Docker container to give access to web UI outside the container. By visiting the host IP address and port, you can redirect to the web UI process started in Docker container and visit web UI content.

For example, you could start a new Docker container from the following command:

.. code-block:: bash

   docker run -i -t -p [hostPort]:[containerPort] [image]

``-i:`` Start a Docker in an interactive mode.

``-t:`` Docker assign the container an input terminal.

``-p:`` Port mapping, map host port to a container port.

For more information about Docker commands, please `refer to this <https://docs.docker.com/v17.09/edge/engine/reference/run/>`__.

Note:

.. code-block:: bash

      NNI only supports Ubuntu and MacOS systems in local mode for the moment, please use correct Docker image type. If you want to use gpu in a Docker container, please use nvidia-docker.

Step 3: Run NNI in a Docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you start a Docker image using NNI's official image ``msranni/nni``\ , you can directly start NNI experiments by using the ``nnictl`` command. Our official image has NNI's running environment and basic python and deep learning frameworks preinstalled.

If you start your own Docker image, you may need to install the NNI package first; please refer to `NNI installation <InstallationLinux.rst>`__.

If you want to run NNI's official examples, you may need to clone the NNI repo in GitHub using

.. code-block:: bash

   git clone https://github.com/Microsoft/nni.git

then you can enter ``nni/examples/trials`` to start an experiment.

After you prepare NNI's environment, you can start a new experiment using the ``nnictl`` command. `See here <QuickStart.rst>`__.

Using Docker on a remote platform
---------------------------------

NNI supports starting experiments in `remoteTrainingService <../TrainingService/RemoteMachineMode.rst>`__\ , and running trial jobs on remote machines. As Docker can start an independent Ubuntu system as an SSH server, a Docker container can be used as the remote machine in NNI's remote mode.

Step 1: Setting a Docker environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You should install the Docker software on your remote machine first, please `refer to this <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`__.

To make sure your Docker container can be connected by NNI experiments, you should build your own Docker image to set an SSH server or use images with an SSH configuration. If you want to use a Docker container as an SSH server, you should configure the SSH password login or private key login; please `refer to this <https://docs.docker.com/engine/examples/running_ssh_service/>`__.

Note:

.. code-block:: text

   NNI's official image msranni/nni does not support SSH servers for the time being; you should build your own Docker image with an SSH configuration or use other images as a remote server.

Step 2: Start a Docker container on a remote machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An SSH server needs a port; you need to expose Docker's SSH port to NNI as the connection port. For example, if you set your container's SSH port as ``A``, you should map the container's port ``A`` to your remote host machine's other port ``B``, NNI will connect port ``B`` as an SSH port, and your host machine will map the connection from port ``B`` to port ``A`` then NNI could connect to your Docker container.

For example, you could start your Docker container using the following commands:

.. code-block:: bash

   docker run -dit -p [hostPort]:[containerPort] [image]

The ``containerPort`` is the SSH port used in your Docker container and the ``hostPort`` is your host machine's port exposed to NNI. You can set your NNI's config file to connect to ``hostPort`` and the connection will be transmitted to your Docker container.
For more information about Docker commands, please `refer to this <https://docs.docker.com/v17.09/edge/engine/reference/run/>`__.

Note:

.. code-block:: bash

   If you use your own Docker image as a remote server, please make sure that this image has a basic python environment and an NNI SDK runtime environment. If you want to use a GPU in a Docker container, please use nvidia-docker.

Step 3: Run NNI experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can set your config file as a remote platform and set the ``machineList`` configuration to connect to your Docker SSH server; `refer to this <../TrainingService/RemoteMachineMode.rst>`__. Note that you should set the correct ``port``\ , ``username``\ , and ``passWd`` or ``sshKeyPath`` of your host machine.

``port:`` The host machine's port, mapping to Docker's SSH port.

``username:`` The username of the Docker container.

``passWd:`` The password of the Docker container.

``sshKeyPath:`` The path of the private key of the Docker container.

After the configuration of the config file, you could start an experiment, `refer to this <QuickStart.rst>`__.
