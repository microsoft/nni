Install NNI
===========

NNI requires Python >= 3.7.
It is tested and supported on Ubuntu >= 18.04,
Windows 10 >= 21H2, and macOS >= 11.

There are 3 ways to install NNI:

* :ref:`Using pip <installation-pip>`
* :ref:`Build source code <installation-source>`
* :ref:`Using Docker <installation-docker>`

.. _installation-pip:

Using pip
---------

NNI provides official packages for x86-64 CPUs. They can be installed with pip:

.. code-block:: text

    pip install nni

Or to upgrade to latest version:

.. code-block:: text

    pip install --latest nni

You can check installation with:

.. code-block:: text

    nnictl --version

On Linux systems without Conda, you may encounter ``bash: nnictl: command not found`` error.
In this case you need to add pip script directory to ``PATH``:

.. code-block:: bash

    echo 'export PATH=${PATH}:${HOME}/.local/bin' >> ~/.bashrc
    source ~/.bashrc

.. _installation-source:

Installing from Source Code
---------------------------

NNI hosts source code on `GitHub <https://github.com/microsoft/nni>`__.

NNI has experimental support for ARM64 CPUs, including Apple M1.
It requires to install from source code.

See :doc:`/notes/build_from_source`.

.. _installation-docker:

Using Docker
------------

NNI provides official Docker image on `Docker Hub <https://hub.docker.com/r/msranni/nni>`__.

.. code-block:: text

    docker pull msranni/nni

Installing Extra Dependencies
-----------------------------

Some built-in algorithms of NNI requires extra packages.
Use ``nni[<algorithm-name>]`` to install their dependencies.

For example, to install dependencies of :class:`DNGO tuner<nni.algorithms.hpo.dngo_tuner.DNGOTuner>` :

.. code-block:: text

    pip install nni[DNGO]

This command will not reinstall NNI itself, even if it was installed in development mode.

Alternatively, you may install all extra dependencies at once:

.. code-block:: text

    pip install nni[all]

**NOTE**: SMAC tuner depends on swig3, which requires a manual downgrade on Ubuntu:

.. code-block:: bash

    sudo apt install swig3.0
    sudo rm /usr/bin/swig
    sudo ln -s swig3.0 /usr/bin/swig
