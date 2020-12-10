Install on Linux & Mac
======================

Installation
------------

Installation on Linux and macOS follow the same instructions, given below.

Install NNI through pip
^^^^^^^^^^^^^^^^^^^^^^^

  Prerequisite: ``python 64-bit >= 3.6``

.. code-block:: bash

     python3 -m pip install --upgrade nni

Install NNI through source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  If you are interested in special or the latest code versions, you can install NNI through source code.

  Prerequisites: ``python 64-bit >=3.6``\ , ``git``\ , ``wget``

.. code-block:: bash

     git clone -b v1.9 https://github.com/Microsoft/nni.git
     cd nni
     ./install.sh

Use NNI in a docker image
^^^^^^^^^^^^^^^^^^^^^^^^^

  You can also install NNI in a docker image. Please follow the instructions :githublink:`here <deployment/docker/README.rst>` to build an NNI docker image. The NNI docker image can also be retrieved from Docker Hub through the command ``docker pull msranni/nni:latest``.

Verify installation
-------------------

The following example is built on TensorFlow 1.x. Make sure **TensorFlow 1.x is used** when running it.


* 
  Download the examples via cloning the source code.

  .. code-block:: bash

     git clone -b v1.9 https://github.com/Microsoft/nni.git

* 
  Run the MNIST example.

  .. code-block:: bash

     nnictl create --config nni/examples/trials/mnist-tfv1/config.yml

* 
  Wait for the message ``INFO: Successfully started experiment!`` in the command line. This message indicates that your experiment has been successfully started. You can explore the experiment using the ``Web UI url``.

.. code-block:: text

   INFO: Starting restful server...
   INFO: Successfully started Restful server!
   INFO: Setting local config...
   INFO: Successfully set local config!
   INFO: Starting experiment...
   INFO: Successfully started experiment!
   -----------------------------------------------------------------------
   The experiment id is egchD4qy
   The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
   -----------------------------------------------------------------------

   You can use these commands to get more information about the experiment
   -----------------------------------------------------------------------
            commands                       description
   1. nnictl experiment show        show the information of experiments
   2. nnictl trial ls               list all of trial jobs
   3. nnictl top                    monitor the status of running experiments
   4. nnictl log stderr             show stderr log content
   5. nnictl log stdout             show stdout log content
   6. nnictl stop                   stop an experiment
   7. nnictl trial kill             kill a trial job by id
   8. nnictl --help                 get help information about nnictl
   -----------------------------------------------------------------------


* Open the ``Web UI url`` in your browser, you can view detailed information about the experiment and all the submitted trial jobs as shown below. `Here <../Tutorial/WebUI.rst>`__ are more Web UI pages.


.. image:: ../../img/webui_overview_page.png
   :target: ../../img/webui_overview_page.png
   :alt: overview



.. image:: ../../img/webui_trialdetail_page.png
   :target: ../../img/webui_trialdetail_page.png
   :alt: detail


System requirements
-------------------

Due to potential programming changes, the minimum system requirements of NNI may change over time.

Linux
^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 
     - Recommended
     - Minimum
   * - **Operating System**
     - Ubuntu 16.04 or above
     -
   * - **CPU**
     - Intel® Core™ i5 or AMD Phenom™ II X3 or better
     - Intel® Core™ i3 or AMD Phenom™ X3 8650
   * - **GPU**
     - NVIDIA® GeForce® GTX 660 or better
     - NVIDIA® GeForce® GTX 460
   * - **Memory**
     - 6 GB RAM
     - 4 GB RAM
   * - **Storage**
     - 30 GB available hare drive space
     -
   * - **Internet**
     - Boardband internet connection
     -
   * - **Resolution**
     - 1024 x 768 minimum display resolution
     -


macOS
^^^^^

.. list-table::
   :header-rows: 1
   :widths: auto

   * -
     - Recommended
     - Minimum
   * - **Operating System**
     - macOS 10.14.1 or above
     - 
   * - **CPU**
     - Intel® Core™ i7-4770 or better
     - Intel® Core™ i5-760 or better
   * - **GPU**
     - AMD Radeon™ R9 M395X or better
     - NVIDIA® GeForce® GT 750M or AMD Radeon™ R9 M290 or better
   * - **Memory**
     - 8 GB RAM
     - 4 GB RAM
   * - **Storage**
     - 70GB available space SSD
     - 70GB available space 7200 RPM HDD
   * - **Internet**
     - Boardband internet connection
     - 
   * - **Resolution**
     - 1024 x 768 minimum display resolution
     - 


Further reading
---------------


* `Overview <../Overview.rst>`__
* `Use command line tool nnictl <Nnictl.rst>`__
* `Use NNIBoard <WebUI.rst>`__
* `Define search space <SearchSpaceSpec.rst>`__
* `Config an experiment <ExperimentConfig.rst>`__
* `How to run an experiment on local (with multiple GPUs)? <../TrainingService/LocalMode.rst>`__
* `How to run an experiment on multiple machines? <../TrainingService/RemoteMachineMode.rst>`__
* `How to run an experiment on OpenPAI? <../TrainingService/PaiMode.rst>`__
* `How to run an experiment on Kubernetes through Kubeflow? <../TrainingService/KubeflowMode.rst>`__
* `How to run an experiment on Kubernetes through FrameworkController? <../TrainingService/FrameworkControllerMode.rst>`__
* `How to run an experiment on Kubernetes through AdaptDL? <../TrainingService/AdaptDLMode.rst>`__
