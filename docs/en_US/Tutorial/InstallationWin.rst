Install on Windows
==================

Prerequires
-----------


* 
  Python 3.6 (or above) 64-bit. `Anaconda <https://www.anaconda.com/products/individual>`__ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ is highly recommended to manage multiple Python environments on Windows.

* 
  If it's a newly installed Python environment, it needs to install `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`__ to support build NNI dependencies like ``scikit-learn``.

  .. code-block:: bat

       pip install cython wheel

* 
  git for verifying installation.

Install NNI
-----------

In most cases, you can install and upgrade NNI from pip package. It's easy and fast.

If you are interested in special or the latest code versions, you can install NNI through source code.

If you want to contribute to NNI, refer to `setup development environment <SetupNniDeveloperEnvironment.rst>`__.


* 
  From pip package

  .. code-block:: bat

       python -m pip install --upgrade nni

* 
  From source code

  .. code-block:: bat

       git clone -b v1.9 https://github.com/Microsoft/nni.git
       cd nni
       powershell -ExecutionPolicy Bypass -file install.ps1

Verify installation
-------------------

The following example is built on TensorFlow 1.x. Make sure **TensorFlow 1.x is used** when running it.


* 
  Clone examples within source code.

  .. code-block:: bat

       git clone -b v1.9 https://github.com/Microsoft/nni.git

* 
  Run the MNIST example.

  .. code-block:: bat

       nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml

    Note:  If you are familiar with other frameworks, you can choose corresponding example under ``examples\trials``. It needs to change trial command ``python3`` to ``python`` in each example YAML, since default installation has ``python.exe``\ , not ``python3.exe`` executable.

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

Below are the minimum system requirements for NNI on Windows, Windows 10.1809 is well tested and recommend. Due to potential programming changes, the minimum system requirements for NNI may change over time.

.. list-table::
   :header-rows: 1
   :widths: auto

   * -
     - Recommended
     - Minimum
   * - **Operating System**
     - Windows 10 1809 or above
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


FAQ
---

simplejson failed when installing NNI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure a C++ 14.0 compiler is installed.

..

   building 'simplejson._speedups' extension error: [WinError 3] The system cannot find the path specified


Trial failed with missing DLL in command line or PowerShell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error is caused by missing LIBIFCOREMD.DLL and LIBMMD.DLL and failure to install SciPy. Using Anaconda or Miniconda with Python(64-bit) can solve it.

..

   ImportError: DLL load failed


Trial failed on webUI
^^^^^^^^^^^^^^^^^^^^^

Please check the trial log file stderr for more details.

If there is a stderr file, please check it. Two possible cases are:


* forgetting to change the trial command ``python3`` to ``python`` in each experiment YAML.
* forgetting to install experiment dependencies such as TensorFlow, Keras and so on.

Fail to use BOHB on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure a C++ 14.0 compiler is installed when trying to run ``nnictl package install --name=BOHB`` to install the dependencies.

Not supported tuner on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SMAC is not supported currently; for the specific reason refer to this `GitHub issue <https://github.com/automl/SMAC3/issues/483>`__.

Use Windows as a remote worker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Refer to `Remote Machine mode <../TrainingService/RemoteMachineMode.rst>`__.

Segmentation fault (core dumped) when installing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Refer to `FAQ <FAQ.rst>`__.

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
