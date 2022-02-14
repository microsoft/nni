Contribution Guide
==================

Great! We are always on the lookout for more contributors to our code base.

Firstly, if you are unsure or afraid of anything, just ask or submit the issue or pull request anyways. You won't be yelled at for giving your best effort. The worst that can happen is that you'll be politely asked to change something. We appreciate any sort of contributions and don't want a wall of rules to get in the way of that.

However, for those individuals who want a bit more guidance on the best way to contribute to the project, read on. This document will cover all the points we're looking for in your contributions, raising your chances of quickly merging or addressing your contributions.

There are a few simple guidelines that you need to follow before providing your hacks.

Bug Reports and Feature Requests
--------------------------------

If you encountered a problem when using NNI, or have an idea for a new feature, please submit it to the `issue tracker <https://github.com/microsoft/nni/issues>`_ on GitHub.

For bug reports, please specify the following details so that our maintainers can help resolve the issue:

* Setup details needs to be filled as specified in the issue template.
* A scenario where the issue occurred (with details on how to reproduce it).
* Errors and log messages that are displayed by the software.
* Any other details that might be useful.

Writing code
------------

There is always something more that is required, to make it easier to suit your use-cases.
Before starting to write code, we recommend checking for `issues <https://github.com/microsoft/nni/issues>`_ on GitHub or open a new issue to initiate a discussion. There could be cases where people are already working on a fix, or similar features have already been under discussion.

To contribute code, you first need to find the NNI code repo located on `GitHub <https://github.com/microsoft/nni>`_. Firstly, fork the repository under your own GitHub handle. After cloning the repository, add, commit, push and squash (if necessary) the changes with detailed commit messages to your fork. From where you can proceed to making a pull request. The pull request will then be reviewed by our core maintainers before merging into master branch. `Here <https://github.com/firstcontributions/first-contributions>`_ is a step-by-step guide for this process.

Find the code snippet that concerns you
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NNI repository is large code-base. High-level speaking, it can be decomposed into several core parts:

* ``nni``: the core Python package that contains most features of hyper-parameter tuner, neural architecture search, model compression.
* ``ts``: contains ``nni_manager`` that manages experiments and training services, and ``webui`` for visualization.
* ``pipelines`` and ``test``: unit test and integration test, alongside their configurations.

See :doc:`./architecture_overview` if you are interested in details.

Get started with development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NNI development environment supports Ubuntu 1604 (or above), and Windows 10 with Python 3. We recommend using `conda <https://docs.conda.io/>`_ on Windows.

1. Fork the NNI's GitHub repository and clone the forked repository to your machine.

   .. code-block:: bash

      git clone https://github.com/<your_github_handle>/nni.git

2. Create a new working branch. Use any name you like.

   .. code-block:: bash

      cd nni
      git checkout -b feature-xyz

3. Install NNI from source code if you need to modify the source code, and test it.

   .. code-block:: bash

      python3 -m pip install -U -r dependencies/setup.txt
      python3 -m pip install -r dependencies/develop.txt
      python3 setup.py develop

   This installs NNI in `development mode <https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html>`_,
   so you don't need to reinstall it after edit.

4. Try to start an experiment to check if your environment is ready. For example, run the command

   .. code-block:: bash

      nnictl create --config examples/trials/mnist-pytorch/config.yml

   And open WebUI to check if everything is OK. Or check the version of installed NNI,

   .. code-block:: python

      >>> import nni
      >>> nni.__version__
      '999.dev0'

   .. note:: Please don't run test under the same folder where the NNI repository is located. As the repository is probably also called ``nni``, it could import the wrong ``nni`` package.

5. Write your code along with tests to verify whether the bug is fixed, or the feature works as expected.

6. Reload changes. For Python, nothing needs to be done, because the code is already linked to package folders. For TypeScript on Linux and MacOS,

   * If ``ts/nni_manager`` is changed, run ``yarn watch`` under this folder. It will watch and build code continually. The ``nnictl`` need to be restarted to reload NNI manager.
   * If ``ts/webui`` is changed, run ``yarn dev``\ , which will run a mock API server and a webpack dev server simultaneously. Use ``EXPERIMENT`` environment variable (e.g., ``mnist-tfv1-running``\ ) to specify the mock data being used. Built-in mock experiments are listed in ``src/webui/mock``. An example of the full command is ``EXPERIMENT=mnist-tfv1-running yarn dev``.

   For TypeScript on Windows, currently you must rebuild TypeScript modules with `python3 setup.py build_ts` after edit.

7. Commit and push your changes, and submit your pull request!

Contributing to Source Code and Bug Fixes
-----------------------------------------

Provide PRs with appropriate tags for bug fixes or enhancements to the source code. Do follow the correct naming conventions and code styles when you work on and do try to implement all code reviews along the way.

If you are looking for How to develop and debug the NNI source code, you can refer to `How to set up NNI developer environment doc <./SetupNniDeveloperEnvironment.rst>`__ file in the ``docs`` folder.

Similarly for `Quick Start <QuickStart.rst>`__. For everything else, refer to `NNI Home page <http://nni.readthedocs.io>`__.

Solve Existing Issues
---------------------

Head over to `issues <https://github.com/Microsoft/nni/issues>`__ to find issues where help is needed from contributors. You can find issues tagged with 'good-first-issue' or 'help-wanted' to contribute in.

A person looking to contribute can take up an issue by claiming it as a comment/assign their Github ID to it. In case there is no PR or update in progress for a week on the said issue, then the issue reopens for anyone to take up again. We need to consider high priority issues/regressions where response time must be a day or so.

Code Styles & Naming Conventions
--------------------------------

* We follow `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ for Python code and naming conventions, do try to adhere to the same when making a pull request or making a change. One can also take the help of linters such as ``flake8`` or ``pylint``
* We also follow `NumPy Docstring Style <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy>`__ for Python Docstring Conventions. During the `documentation building <Contributing.rst#documentation>`__\ , we use `sphinx.ext.napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`__ to generate Python API documentation from Docstring.
* For docstrings, please refer to `numpydoc docstring guide <https://numpydoc.readthedocs.io/en/latest/format.html>`__ and `pandas docstring guide <https://python-sprints.github.io/pandas/guide/pandas_docstring.html>`__

  * For function docstring, **description**, **Parameters**, and **Returns** **Yields** are mandatory.
  * For class docstring, **description**, **Attributes** are mandatory.
  * For docstring to describe ``dict``, which is commonly used in our hyper-param format description, please refer to `Internal Guideline on Writing Standards <https://ribokit.github.io/docs/text/>`__

Documentation
-------------

Our documentation is built with :githublink:`sphinx <docs>`.

* Before submitting the documentation change, please **build homepage locally**: ``cd docs/en_US && make html``, then you can see all the built documentation webpage under the folder ``docs/en_US/_build/html``. It's also highly recommended taking care of **every WARNING** during the build, which is very likely the signal of a **deadlink** and other annoying issues.

* 
  For links, please consider using **relative paths** first. However, if the documentation is written in reStructuredText format, and:


  * It's an image link which needs to be formatted with embedded html grammar, please use global URL like ``https://user-images.githubusercontent.com/44491713/51381727-e3d0f780-1b4f-11e9-96ab-d26b9198ba65.png``, which can be automatically generated by dragging picture onto `Github Issue <https://github.com/Microsoft/nni/issues/new>`__ Box.
  * It cannot be re-formatted by sphinx, such as source code, please use its global URL. For source code that links to our github repo, please use URLs rooted at ``https://github.com/Microsoft/nni/tree/master/`` (:githublink:`mnist.py <examples/trials/mnist-pytorch/mnist.py>` for example).


https://docutils.sourceforge.io/docs/user/rst/quickstart.html
https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html


Setup NNI development environment
=================================

NNI development environment supports Ubuntu 1604 (or above), and Windows 10 with Python3 64bit.

1. Clone source code
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/Microsoft/nni.git

Note, if you want to contribute code back, it needs to fork your own NNI repo, and clone from there.

2. Install from source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3 -m pip install -U -r dependencies/setup.txt
   python3 -m pip install -r dependencies/develop.txt
   python3 setup.py develop

This installs NNI in `development mode <https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html>`__,
so you don't need to reinstall it after edit.

3. Check if the environment is ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, you can try to start an experiment to check if your environment is ready.
For example, run the command

.. code-block:: bash

   nnictl create --config examples/trials/mnist-pytorch/config.yml

And open WebUI to check if everything is OK

4. Reload changes
^^^^^^^^^^^^^^^^^

Python
******

Nothing to do, the code is already linked to package folders.

TypeScript (Linux and macOS)
****************************

* If ``ts/nni_manager`` is changed, run ``yarn watch`` under this folder. It will watch and build code continually. The ``nnictl`` need to be restarted to reload NNI manager.
* If ``ts/webui`` is changed, run ``yarn dev``\ , which will run a mock API server and a webpack dev server simultaneously. Use ``EXPERIMENT`` environment variable (e.g., ``mnist-tfv1-running``\ ) to specify the mock data being used. Built-in mock experiments are listed in ``src/webui/mock``. An example of the full command is ``EXPERIMENT=mnist-tfv1-running yarn dev``.

TypeScript (Windows)
********************

Currently you must rebuild TypeScript modules with `python3 setup.py build_ts` after edit.

5. Submit Pull Request
^^^^^^^^^^^^^^^^^^^^^^

All changes are merged to master branch from your forked repo. The description of Pull Request must be meaningful, and useful.

We will review the changes as soon as possible. Once it passes review, we will merge it to master branch.

For more contribution guidelines and coding styles, you can refer to the `contributing document <Contributing.rst>`__.
