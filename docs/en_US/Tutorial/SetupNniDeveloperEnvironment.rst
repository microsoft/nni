Setup NNI development environment
=================================

NNI development environment supports Ubuntu 1604 (or above), and Windows 10 with Python3 64bit.

Installation
------------

The installation steps are similar with installing from source code. But the installation links to code directory, so that code changes can be applied to installation as easy as possible.

1. Clone source code
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Microsoft/nni.git

Note, if you want to contribute code back, it needs to fork your own NNI repo, and clone from there.

2. Install from source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ubuntu
^^^^^^

.. code-block:: bash

   make dev-easy-install

Windows
^^^^^^^

.. code-block:: bat

   powershell -ExecutionPolicy Bypass -file install.ps1 -Development

3. Check if the environment is ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, you can try to start an experiment to check if your environment is ready.
For example, run the command

.. code-block:: bash

   nnictl create --config examples/trials/mnist-tfv1/config.yml

And open WebUI to check if everything is OK

4. Reload changes
^^^^^^^^^^^^^^^^^

Python
^^^^^^

Nothing to do, the code is already linked to package folders.

TypeScript
^^^^^^^^^^


* If ``src/nni_manager`` is changed, run ``yarn watch`` under this folder. It will watch and build code continually. The ``nnictl`` need to be restarted to reload NNI manager.
* If ``src/webui`` is changed, run ``yarn dev``\ , which will run a mock API server and a webpack dev server simultaneously. Use ``EXPERIMENT`` environment variable (e.g., ``mnist-tfv1-running``\ ) to specify the mock data being used. Built-in mock experiments are listed in ``src/webui/mock``. An example of the full command is ``EXPERIMENT=mnist-tfv1-running yarn dev``.
* If ``src/nasui`` is changed, run ``yarn start`` under the corresponding folder. The web UI will refresh automatically if code is changed. There is also a mock API server that is useful when developing. It can be launched via ``node server.js``.

5. Submit Pull Request
^^^^^^^^^^^^^^^^^^^^^^

All changes are merged to master branch from your forked repo. The description of Pull Request must be meaningful, and useful.

We will review the changes as soon as possible. Once it passes review, we will merge it to master branch.

For more contribution guidelines and coding styles, you can refer to the `contributing document <Contributing.rst>`__.
