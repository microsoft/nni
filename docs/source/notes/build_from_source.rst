Build from Source
=================

This article describes how to build and install NNI from `source code`_.

We recommend using latest setuptools:

.. code-block:: text

    pip install --upgrade setuptools pip wheel

.. _source code: https://github.com/microsoft/nni

Development Build
-----------------

If you want to build NNI for your own use, we recommend using `development mode`_.

.. code-block:: text

    python setup.py develop

This will install NNI as symlink, and the version number will be ``999.dev0``.

.. _development mode: https://setuptools.pypa.io/en/latest/userguide/development_mode.html

Release Build
-------------

To install in release mode, you must first build a wheel.
NNI does not support setuptools' "install" command.

A release package requires jupyterlab to build the extension:

.. code-block:: text

    pip install jupyterlab

And you need to set ``NNI_RELEASE`` environment variable, and compile TypeScript modules before "bdist_wheel".

In bash:

.. code-block:: bash

    export NNI_RELEASE=2.7
    python setup.py build_ts
    python bdist_wheel

In PowerShell:

.. code-block:: powershell

    $env:NNI_RELEASE=2.7
    python setup.py build_ts
    python bdist_wheel

If successful, you will find the wheel in ``dist`` directory.

.. note::

    NNI's build process is somewhat complicated.
    This is due to setuptools and TypeScript not working well together.

    Setuptools require to provide ``package_data``, the full list of package files, before running any command.
    However it is nearly impossible to predict what files will be generated before invoking TypeScript compiler.

    If you have any solution for this problem, please open an issue to let us know.

Build Docker Image
------------------

You can build a Docker image with :githublink:`Dockerfile <Dockerfile>`:

.. code-block:: bash

    export NNI_RELEASE=2.7
    python setup.py build_ts
    python setup.py bdist_wheel -p manylinux1_x86_64
    docker build --build-arg NNI_RELEASE=${NNI_RELEASE} -t my/nni .

To build image for other platforms, please edit Dockerfile yourself.

Other Commands and Options
--------------------------

Clean
^^^^^

If the build fails, please clean up and try again:

.. code:: text

    python setup.py clean

Skip compiling TypeScript modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is useful when you have uninstalled NNI from development mode and want to install again.

It will not work if you have never built TypeScript modules before.

.. code:: text

    python setup.py develop --skip-ts
