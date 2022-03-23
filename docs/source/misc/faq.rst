FAQ
===

This page is for frequent asked questions and answers.

tmp folder fulled
^^^^^^^^^^^^^^^^^

nnictl will use tmp folder as a temporary folder to copy files under codeDir when executing experimentation creation.
When met errors like below, try to clean up **tmp** folder first.

..

   OSError: [Errno 28] No space left on device


Cannot get trials' metrics in OpenPAI mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In OpenPAI training mode, we start a rest server which listens on 51189 port in NNI Manager to receive metrcis reported from trials running in OpenPAI cluster. If you didn't see any metrics from WebUI in OpenPAI mode, check your machine where NNI manager runs on to make sure 51189 port is turned on in the firewall rule.

Segmentation Fault (core dumped) when installing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   make: *** [install-XXX] Segmentation fault (core dumped)


Please try the following solutions in turn:


* Update or reinstall you current python's pip like ``python3 -m pip install -U pip``
* Install NNI with ``--no-cache-dir`` flag like ``python3 -m pip install nni --no-cache-dir``

Job management error: getIPV4Address() failed because os.networkInterfaces().eth0 is undefined.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your machine don't have eth0 device, please set `nniManagerIp <ExperimentConfig.rst>`__ in your config file manually.

Exceed the MaxDuration but didn't stop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the duration of experiment reaches the maximum duration, nniManager will not create new trials, but the existing trials will continue unless user manually stop the experiment.

Could not stop an experiment using ``nnictl stop``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you upgrade your NNI or you delete some config files of NNI when there is an experiment running, this kind of issue may happen because the loss of config file. You could use ``ps -ef | grep node`` to find the PID of your experiment, and use ``kill -9 {pid}`` to kill it manually.

Could not get ``default metric`` in webUI of virtual machines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Config the network mode to bridge mode or other mode that could make virtual machine's host accessible from external machine, and make sure the port of virtual machine is not forbidden by firewall.

Could not open webUI link
^^^^^^^^^^^^^^^^^^^^^^^^^

Unable to open the WebUI may have the following reasons:


* ``http://127.0.0.1``\ , ``http://172.17.0.1`` and ``http://10.0.0.15`` are referred to localhost, if you start your experiment on the server or remote machine. You can replace the IP to your server IP to view the WebUI, like ``http://[your_server_ip]:8080``
* If you still can't see the WebUI after you use the server IP, you can check the proxy and the firewall of your machine. Or use the browser on the machine where you start your NNI experiment.
* Another reason may be your experiment is failed and NNI may fail to get the experiment information. You can check the log of NNIManager in the following directory: ``~/nni-experiments/[your_experiment_id]`` ``/log/nnimanager.log``

Restful server start failed
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Probably it's a problem with your network config. Here is a checklist.


* You might need to link ``127.0.0.1`` with ``localhost``. Add a line ``127.0.0.1 localhost`` to ``/etc/hosts``.
* It's also possible that you have set some proxy config. Check your environment for variables like ``HTTP_PROXY`` or ``HTTPS_PROXY`` and unset if they are set.

NNI on Windows problems
^^^^^^^^^^^^^^^^^^^^^^^

Please refer to `NNI on Windows <InstallationWin.rst>`__

More FAQ issues
^^^^^^^^^^^^^^^

`NNI Issues with FAQ labels <https://github.com/microsoft/nni/labels/FAQ>`__

Help us improve
^^^^^^^^^^^^^^^

Please inquiry the problem in https://github.com/Microsoft/nni/issues to see whether there are other people already reported the problem, create a new one if there are no existing issues been created.
