QuickStart
==========

Installation
------------

We currently support Linux, macOS, and Windows. Ubuntu 16.04 or higher, macOS 10.14.1, and Windows 10.1809 are tested and supported. Simply run the following ``pip install`` in an environment that has ``python >= 3.6``.

Linux and macOS
^^^^^^^^^^^^^^^

.. code-block:: bash

   python3 -m pip install --upgrade nni

Windows
^^^^^^^

.. code-block:: bash

   python -m pip install --upgrade nni

.. Note:: For Linux and macOS, ``--user`` can be added if you want to install NNI in your home directory; this does not require any special privileges.

.. Note:: If there is an error like ``Segmentation fault``, please refer to the :doc:`FAQ <FAQ>`.

.. Note:: For the system requirements of NNI, please refer to :doc:`Install NNI on Linux & Mac <InstallationLinux>` or :doc:`Windows <InstallationWin>`.

Enable NNI Command-line Auto-Completion (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the installation, you may want to enable the auto-completion feature for **nnictl** commands. Please refer to this `tutorial <../CommunitySharings/AutoCompletion.rst>`__.

"Hello World" example on MNIST
------------------------------

NNI is a toolkit to help users run automated machine learning experiments. It can automatically do the cyclic process of getting hyperparameters, running trials, testing results, and tuning hyperparameters. Here, we'll show how to use NNI to help you find the optimal hyperparameters for a MNIST model.

Here is an example script to train a CNN on the MNIST dataset **without NNI**:

.. code-block:: python

    def main(args):
        # load data
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(...), batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.tuils.data.DataLoader(datasets.MNIST(...), batch_size=1000, shuffle=True)
        # build model
        model = Net(hidden_size=args['hidden_size'])
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
        # train
        for epoch in range(10):
            train(args, model, device, train_loader, optimizer, epoch)
            test_acc = test(args, model, device, test_loader)
            print(test_acc)
        print('final accuracy:', test_acc)
         
    if __name__ == '__main__':
        params = {
            'batch_size': 32,
            'hidden_size': 128,
            'lr': 0.001,
            'momentum': 0.5
        }
        main(params)

The above code can only try one set of parameters at a time; if we want to tune learning rate, we need to manually modify the hyperparameter and start the trial again and again.

NNI is born to help the user do tuning jobs; the NNI working process is presented below:

.. code-block:: text

   input: search space, trial code, config file
   output: one optimal hyperparameter configuration

   1: For t = 0, 1, 2, ..., maxTrialNum,
   2:      hyperparameter = chose a set of parameter from search space
   3:      final result = run_trial_and_evaluate(hyperparameter)
   4:      report final result to NNI
   5:      If reach the upper limit time,
   6:          Stop the experiment
   7: return hyperparameter value with best final result

If you want to use NNI to automatically train your model and find the optimal hyper-parameters, you need to do three changes based on your code:

Three steps to start an experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Step 1**: Write a ``Search Space`` file in JSON, including the ``name`` and the ``distribution`` (discrete-valued or continuous-valued) of all the hyperparameters you need to search.

.. code-block:: diff

    -   params = {'batch_size': 32, 'hidden_size': 128, 'lr': 0.001, 'momentum': 0.5}
    +   {
    +       "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    +       "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
    +       "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
    +       "momentum":{"_type":"uniform","_value":[0, 1]}
    +   }

*Example:* :githublink:`search_space.json <examples/trials/mnist-pytorch/search_space.json>`

**Step 2**\ : Modify your ``Trial`` file to get the hyperparameter set from NNI and report the final result to NNI.

.. code-block:: diff

    + import nni

      def main(args):
          # load data
          train_loader = torch.utils.data.DataLoader(datasets.MNIST(...), batch_size=args['batch_size'], shuffle=True)
          test_loader = torch.tuils.data.DataLoader(datasets.MNIST(...), batch_size=1000, shuffle=True)
          # build model
          model = Net(hidden_size=args['hidden_size'])
          optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
          # train
          for epoch in range(10):
              train(args, model, device, train_loader, optimizer, epoch)
              test_acc = test(args, model, device, test_loader)
    -         print(test_acc)
    +         nni.report_intermeidate_result(test_acc)
    -     print('final accuracy:', test_acc)
    +     nni.report_final_result(test_acc)
           
      if __name__ == '__main__':
    -     params = {'batch_size': 32, 'hidden_size': 128, 'lr': 0.001, 'momentum': 0.5}
    +     params = nni.get_next_parameter()
          main(params)

*Example:* :githublink:`mnist.py <examples/trials/mnist-pytorch/mnist.py>`

**Step 3**\ : Define a ``config`` file in YAML which declares the ``path`` to the search space and trial files. It also gives other information such as the tuning algorithm, max trial number, and max duration arguments.

.. code-block:: yaml

   authorName: default
   experimentName: example_mnist
   trialConcurrency: 1
   maxExecDuration: 1h
   maxTrialNum: 10
   trainingServicePlatform: local
   # The path to Search Space
   searchSpacePath: search_space.json
   useAnnotation: false
   tuner:
     builtinTunerName: TPE
   # The path and the running command of trial
   trial:
     command: python3 mnist.py
     codeDir: .
     gpuNum: 0

.. Note:: If you are planning to use remote machines or clusters as your :doc:`training service <../TrainingService/Overview>`, to avoid too much pressure on network, we limit the number of files to 2000 and total size to 300MB. If your codeDir contains too many files, you can choose which files and subfolders should be excluded by adding a ``.nniignore`` file that works like a ``.gitignore`` file. For more details on how to write this file, see the `git documentation <https://git-scm.com/docs/gitignore#_pattern_format>`__.

*Example:* :githublink:`config.yml <examples/trials/mnist-pytorch/config.yml>` and :githublink:`.nniignore <examples/trials/mnist-pytorch/.nniignore>`

All the code above is already prepared and stored in :githublink:`examples/trials/mnist-pytorch/ <examples/trials/mnist-pytorch>`.

Linux and macOS
^^^^^^^^^^^^^^^

Run the **config.yml** file from your command line to start an MNIST experiment.

.. code-block:: bash

   nnictl create --config nni/examples/trials/mnist-pytorch/config.yml

Windows
^^^^^^^

Run the **config_windows.yml** file from your command line to start an MNIST experiment.

.. code-block:: bash

   nnictl create --config nni\examples\trials\mnist-pytorch\config_windows.yml

.. Note:: If you're using NNI on Windows, you probably need to change ``python3`` to ``python`` in the config.yml file or use the config_windows.yml file to start the experiment.

.. Note:: ``nnictl`` is a command line tool that can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc. Click :doc:`here <Nnictl>` for more usage of ``nnictl``.

Wait for the message ``INFO: Successfully started experiment!`` in the command line. This message indicates that your experiment has been successfully started. And this is what we expect to get:

.. code-block:: text

   INFO: Starting restful server...
   INFO: Successfully started Restful server!
   INFO: Setting local config...
   INFO: Successfully set local config!
   INFO: Starting experiment...
   INFO: Successfully started experiment!
   -----------------------------------------------------------------------
   The experiment id is egchD4qy
   The Web UI urls are: [Your IP]:8080
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

If you prepared ``trial``\ , ``search space``\ , and ``config`` according to the above steps and successfully created an NNI job, NNI will automatically tune the optimal hyper-parameters and run different hyper-parameter sets for each trial according to the requirements you set. You can clearly see its progress through the NNI WebUI.

WebUI
-----

After you start your experiment in NNI successfully, you can find a message in the command-line interface that tells you the ``Web UI url`` like this:

.. code-block:: text

   The Web UI urls are: [Your IP]:8080

Open the ``Web UI url`` (Here it's: ``[Your IP]:8080``\ ) in your browser; you can view detailed information about the experiment and all the submitted trial jobs as shown below. If you cannot open the WebUI link in your terminal, please refer to the `FAQ <FAQ.rst>`__.

View overview page
^^^^^^^^^^^^^^^^^^


Information about this experiment will be shown in the WebUI, including the experiment trial profile and search space message. NNI also supports downloading this information and the parameters through the **Experiment summary** button.


.. image:: ../../img/webui-img/full-oview.png
   :target: ../../img/webui-img/full-oview.png
   :alt: overview



View trials detail page
^^^^^^^^^^^^^^^^^^^^^^^

We could see best trial metrics and hyper-parameter graph in this page. And the table content includes more columns when you click the button ``Add/Remove columns``.


.. image:: ../../img/webui-img/full-detail.png
   :target: ../../img/webui-img/full-detail.png
   :alt: detail



View experiments management page
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the ``All experiments`` page, you can see all the experiments on your machine. 

.. image:: ../../img/webui-img/managerExperimentList/expList.png
   :target: ../../img/webui-img/managerExperimentList/expList.png
   :alt: Experiments list



More detail please refer `the doc <./WebUI.rst>`__.

Related Topic
-------------


* `Launch Tensorboard on WebUI <Tensorboard.rst>`__
* `Try different Tuners <../Tuner/BuiltinTuner.rst>`__
* `Try different Assessors <../Assessor/BuiltinAssessor.rst>`__
* `How to use command line tool nnictl <Nnictl.rst>`__
* `How to write a trial <../TrialExample/Trials.rst>`__
* `How to run an experiment on local (with multiple GPUs)? <../TrainingService/LocalMode.rst>`__
* `How to run an experiment on multiple machines? <../TrainingService/RemoteMachineMode.rst>`__
* `How to run an experiment on OpenPAI? <../TrainingService/PaiMode.rst>`__
* `How to run an experiment on Kubernetes through Kubeflow? <../TrainingService/KubeflowMode.rst>`__
* `How to run an experiment on Kubernetes through FrameworkController? <../TrainingService/FrameworkControllerMode.rst>`__
* `How to run an experiment on Kubernetes through AdaptDL? <../TrainingService/AdaptDLMode.rst>`__
