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

Coding Tips
-----------

We expect all contributors to respect the following coding styles and naming conventions upon their contribution.

Python
^^^^^^

* We follow `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ for Python code and naming conventions, do try to adhere to the same when making a pull request. Our pull request has a mandatory code scan with ``pylint`` and ``flake8``.

  .. note:: To scan your own code locally, run

     .. code-block:: bash

         python -m pylint --rcfile pylintrc nni

  .. tip:: One can also take the help of auto-format tools such as `autopep8 <https://code.visualstudio.com/docs/python/editing#_formatting>`_, which will automatically resolve most of the styling issues.

* We recommend documenting all the methods and classes in your code. Follow `NumPy Docstring Style <https://numpydoc.readthedocs.io/en/latest/format.html>`__ for Python Docstring Conventions.

  * For function docstring, **description**, **Parameters**, and **Returns** are mandatory.
  * For class docstring, **description**, **Attributes** are mandatory. The parameters of ``__init__`` should be documented in the docstring of docs.
  * For docstring to describe ``dict``, which is commonly used in our hyper-parameter format description, please refer to `Internal Guideline on Writing Standards <https://ribokit.github.io/docs/text/>`_.

  .. tip:: `A cheatsheet provided by Sphinx <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy>`__ shows a number of examples of docstring in numpy style.

TypeScript
^^^^^^^^^^

TypeScript code checks can be done with,

.. code-block:: bash

   # for nni manager
   cd ts/nni_manager
   yarn eslint

   # for webui
   cd ts/webui
   yarn sanity-check

Tests
-----

When a new feature is added or a bug is fixed, tests are highly recommended to make sure that the fix is effective or the feature won't break in future. There are two types of tests in NNI:

* Unit test (**UT**): each test targets at a specific class / function / module.
* Integration test (**IT**): each test is an end-to-end example / demo.

Unit test (Python)
^^^^^^^^^^^^^^^^^^



Unit test (TypeScript)
^^^^^^^^^^^^^^^^^^^^^^

TypeScript UT are paired with TypeScript code. Use ``yarn test`` to run them.

Integration test
^^^^^^^^^^^^^^^^

The integration tests can be found in ``pipelines/`` folder. 



Documentation
-------------

Build and check documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our documentation is located under ``docs/`` folder. The following command can be used to build the documentation.

.. code-block:: bash

   cd docs
   make html

It's also highly recommended taking care of **every WARNING** during the build, which is very likely the signal of a **deadlink** and other annoying issues. Our code check will also make sure that the documentation build completes with no warning.

The built documentation can be found in ``docs/build/html`` folder.

.. attention:: Always use your web browser to check the documentation before committing your change.

.. tip:: `Live Server <https://github.com/ritwickdey/vscode-live-server>`_ is a great extension if you are looking for a static-files server to serve contents in ``docs/build/html``.


Writing new documents
^^^^^^^^^^^^^^^^^^^^^

`ReStructuredText <https://docutils.sourceforge.io/docs/user/rst/quickstart.html>`_ is our documentation language. Sphinx has `an excellent cheatsheet of rst <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ which contains almost everything you might need to know to write a elegant document.

Other than built-in directives provided by Sphinx, we also provide some custom directives:

* ``.. cardlinkitem::``: A tutorial card, useful in :doc:`../tutorial`.
* ``:githublink:`path/to/file.ext` `` or ``:githublink:`text <path/to/file.ext>` ``: reference a file on the GitHub. Linked to the same commit id as where the documentation is built.

Writing new tutorials
^^^^^^^^^^^^^^^^^^^^^


Chinese translation
^^^^^^^^^^^^^^^^^^^
