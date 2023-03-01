Contribution Guide
==================

Great! We are always on the lookout for more contributors to our code base.

Firstly, if you are unsure or afraid of anything, just ask or submit the issue or pull request anyways. You won't be yelled at for giving your best effort. The worst that can happen is that you'll be politely asked to change something. We appreciate any sort of contributions and don't want a wall of rules to get in the way of that.

However, for those individuals who want a bit more guidance on the best way to contribute to the project, read on. This document will cover all the points we're looking for in your contributions, raising your chances of quickly merging or addressing your contributions.

There are a few simple guidelines that you need to follow before providing your hacks.

Bug Reports and Feature Requests
--------------------------------

If you encountered a problem when using NNI, or have an idea for a new feature, your feedbacks are always welcome. Here are some possible channels:

*  `File an issue <https://github.com/microsoft/nni/issues/new/choose>`_ on GitHub.
*  Open or participate in a `discussion <https://github.com/microsoft/nni/discussions>`_.
*  Discuss on the NNI `Gitter <https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge>`_ in NNI.
*  Join IM discussion groups:

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - Gitter
        - WeChat
      * - .. image:: https://user-images.githubusercontent.com/39592018/80665738-e0574a80-8acc-11ea-91bc-0836dc4cbf89.png
        - .. image:: https://github.com/scarlett2018/nniutil/raw/master/wechat.png

Looking for an existing issue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you create a new issue, please do a search in `open issues <https://github.com/microsoft/nni/issues>`_ to see if the issue or feature request has already been filed.

Be sure to scan through the `most popular <https://github.com/microsoft/nni/issues?q=is%3Aopen+is%3Aissue+label%3AFAQ+sort%3Areactions-%2B1-desc>`_ feature requests.

If you find your issue already exists, make relevant comments and add your `reaction <https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments>`_. Use a reaction in place of a "+1" comment:

* 👍 - upvote
* 👎 - downvote

If you cannot find an existing issue that describes your bug or feature, create a new issue following the guidelines below.

Writing good bug reports or feature requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* File a single issue per problem and feature request. Do not enumerate multiple bugs or feature requests in the same issue.

* Provide as much information as you think might relevant to the context (thinking the issue is assigning to you, what kinds of info you will need to debug it!!!). To give you a general idea about what kinds of info are useful for developers to dig out the issue, we had provided issue template for you.

* Once you had submitted an issue, be sure to follow it for questions and discussions. 

* Once the bug is fixed or feature is addressed, be sure to close the issue.

Writing code
------------

There is always something more that is required, to make it easier to suit your use-cases.
Before starting to write code, we recommend checking for `issues <https://github.com/microsoft/nni/issues>`_ on GitHub or open a new issue to initiate a discussion. There could be cases where people are already working on a fix, or similar features have already been under discussion.

To contribute code, you first need to find the NNI code repo located on `GitHub <https://github.com/microsoft/nni>`_. Firstly, fork the repository under your own GitHub handle. After cloning the repository, add, commit, push and squash (if necessary) the changes with detailed commit messages to your fork. From where you can proceed to making a pull request. The pull request will then be reviewed by our core maintainers before merging into master branch. `Here <https://github.com/firstcontributions/first-contributions>`_ is a step-by-step guide for this process.

Contributions to NNI should follow our code of conduct. Please see details :ref:`here <code-of-conduct>`.

Find the code snippet that concerns you
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NNI repository is large code-base. High-level speaking, it can be decomposed into several core parts:

* ``nni``: the core Python package that contains most features of hyper-parameter tuner, neural architecture search, model compression.
* ``ts``: contains ``nni_manager`` that manages experiments and training services, and ``webui`` for visualization.
* ``pipelines`` and ``test``: unit test and integration test, alongside their configurations.

See :doc:`./architecture_overview` if you are interested in details.

.. _get-started-dev:

Get started with development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NNI development environment supports Ubuntu 1604 (or above), and Windows 10 with Python 3.7+ (documentation build requires Python 3.8+). We recommend using `conda <https://docs.conda.io/>`_ on Windows.

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
  * For class docstring, **description** is mandatory. Optionally **Parameters** and **Attributes**. The parameters of ``__init__`` should be documented in the docstring of class.
  * For docstring to describe ``dict``, which is commonly used in our hyper-parameter format description, please refer to `Internal Guideline on Writing Standards <https://ribokit.github.io/docs/text/>`_.

  .. tip:: Basically, you can use :ref:`ReStructuredText <restructuredtext-intro>` syntax in docstrings, without some exceptions. For example, custom headings are not allowed in docstrings.

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

Python UT are located in ``test/ut/`` folder. We use `pytest <https://docs.pytest.org/>`_ to launch the tests, and the working directory is ``test/ut/``.

.. tip:: pytest can be used on a single file or a single test function.

   .. code-block:: bash

      pytest sdk/test_tuner.py
      pytest sdk/test_tuner.py::test_tpe

Unit test (TypeScript)
^^^^^^^^^^^^^^^^^^^^^^

TypeScript UT are paired with TypeScript code. Use ``yarn test`` to run them.

Integration test
^^^^^^^^^^^^^^^^

The integration tests can be found in ``pipelines/`` folder. 

The integration tests are run on Azure DevOps platform on a daily basis, in order to make sure that our examples and training service integrations work properly. However, for critical changes that have impacts on the core functionalities of NNI, we recommend to `trigger the pipeline on the pull request branch <https://stackoverflow.com/questions/60157818/azure-pipeline-run-build-on-pull-request-branch>`_.

The integration tests won't be automatically triggered on pull requests. You might need to contact the core developers to help you trigger the tests.

Documentation
-------------

Build and check documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our documentation is located under ``docs/`` folder. The following command can be used to build the documentation.

.. code-block:: bash

   cd docs
   make en

.. note::

   If you experience issues in building documentation, and see errors like:
      
   * ``Could not import extension xxx (exception: No module named 'xxx')`` : please check your development environment and make sure dependencies have been properly installed: :ref:`get-started-dev`.
   * ``unsupported pickle protocol: 5``: please upgrade to Python 3.8.
   * ``autodoc: No module named 'xxx'``: some dependencies in ``dependencies/`` are not installed. In this case, documentation can be still mostly successfully built, but some API reference could be missing.

It's also highly recommended taking care of **every WARNING** during the build, which is very likely the signal of a **deadlink** and other annoying issues. Our code check will also make sure that the documentation build completes with no warning.

The built documentation can be found in ``docs/build/html`` folder.

.. attention:: Always use your web browser to check the documentation before committing your change.

.. tip:: `Live Server <https://github.com/ritwickdey/vscode-live-server>`_ is a great extension if you are looking for a static-files server to serve contents in ``docs/build/html``.

Writing new documents
^^^^^^^^^^^^^^^^^^^^^

.. |link_example| raw:: html

   <code class="docutils literal notranslate">`Link text &lt;https://domain.invalid/&gt;`_</code>

.. |link_example_2| raw:: html

   <code class="docutils literal notranslate">`Link text &lt;https://domain.invalid/&gt;`__</code>

.. |link_example_3| raw:: html

   <code class="docutils literal notranslate">:doc:`./relative/to/my_doc`</code>

.. |githublink_example| raw:: html

   <code class="docutils literal notranslate">:githublink:`path/to/file.ext`</code>

.. |githublink_example_2| raw:: html

   <code class="docutils literal notranslate">:githublink:`text &lt;path/to/file.ext&gt;`</code>

.. _restructuredtext-intro:

`ReStructuredText <https://docutils.sourceforge.io/docs/user/rst/quickstart.html>`_ is our documentation language. Please find the reference of RST `here <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html>`__.

.. tip:: Sphinx has `an excellent cheatsheet of rst <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ which contains almost everything you might need to know to write a elegant document.

**Dealing with sections.** ``=`` for sections. ``-`` for subsections. ``^`` for subsubsections. ``"`` for paragraphs.

**Dealing with images.** Images should be put into ``docs/img`` folder. Then, reference the image in the document with relative links. For example, ``.. image:: ../../img/example.png``.

**Dealing with codes.** We recommend using ``.. code-block:: python`` to start a code block. The ``python`` here annotates the syntax highlighting.

**Dealing with links.** Use |link_example_3| for links to another doc (no suffix like ``.rst``). To reference a specific section, please use ``:ref:`` (see `Cross-referencing arbitrary locations <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#cross-referencing-arbitrary-locations>`_). For general links that ``:doc:`` and ``:ref:`` can't handle, you can also use |link_example| for inline web links. Note that use one underline might cause `"duplicated target name" error <https://stackoverflow.com/questions/27420317/restructured-text-rst-http-links-underscore-vs-use>`_ when multiple targets share the same name. In that case, use double-underline to avoid the error: |link_example_2|.

Other than built-in directives provided by Sphinx, we also provide some custom directives:

* ``.. cardlinkitem::``: A tutorial card, useful in :doc:`/examples`.
* |githublink_example| or |githublink_example_2|: reference a file on the GitHub. Linked to the same commit id as where the documentation is built.

Writing new tutorials
^^^^^^^^^^^^^^^^^^^^^

Our tutorials are powered by `sphinx-gallery <https://sphinx-gallery.github.io/>`_. Sphinx-gallery is an extension that builds an HTML gallery of examples from any set of Python scripts.

To contribute a new tutorial, here are the steps to follow:

1. Create a notebook styled python file. If you want it executed while inserted into documentation, save the file under ``examples/tutorials/``. If your tutorial contains other auxiliary scripts which are not intended to be included into documentation, save them under ``examples/tutorials/scripts/``.

   .. tip:: The syntax to write a "notebook styled python file" is very simple. In essence, you only need to write a slightly well formatted python file. Here is a useful guide of `how to structure your Python scripts for Sphinx-Gallery <https://sphinx-gallery.github.io/stable/syntax.html>`_.

2. Put the tutorials into ``docs/source/tutorials.rst``. You should add it both in ``toctree`` (to make it appear in the sidebar content table), and ``cardlinkitem`` (to create a card link), and specify the appropriate ``header``, ``description``, ``link``, ``image``, ``background`` (for image) and ``tags``.

   ``link`` are the generated link, which is usually ``tutorials/<your_python_file_name>.html``. Some useful images can be found in ``docs/img/thumbnails``, but you can always use your own. Available background colors are: ``red``, ``pink``, ``purple``, ``deep-purple``, ``blue``, ``light-blue``, ``cyan``, ``teal``, ``green``, ``deep-orange``, ``brown``, ``indigo``.

   In case you prefer to write your tutorial in jupyter, you can use `this script <https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe>`_ to convert the notebook to python file. After conversion and addition to the project, please make sure the sections headings etc are in logical order.

3. Build the tutorials. Since some of the tutorials contain complex AutoML examples, it's very inefficient to build them over and over again. Therefore, we cache the built tutorials in ``docs/source/tutorials``, so that the unchanged tutorials won't be rebuilt. To trigger the build, run ``make en``. This will execute the tutorials and convert the scripts into HTML files. How long it takes depends on your tutorial. As ``make en`` is not very debug-friendly, we suggest making the script runnable by itself before using this building tool.

.. note::

   Some useful HOW-TOs in writing new tutorials:

   * `How to force rebuilding one tutorial <https://sphinx-gallery.github.io/stable/configuration.html#rerunning-stale-examples>`_.
   * `How to add images to notebooks <https://sphinx-gallery.github.io/stable/configuration.html#adding-images-to-notebooks>`_.
   * `How to reference a tutorial in documentation <https://sphinx-gallery.github.io/stable/advanced.html#cross-referencing>`_.

Translation (i18n)
^^^^^^^^^^^^^^^^^^

We only maintain `a partial set of documents <https://github.com/microsoft/nni/issues/4298>`_ with translation. Currently, translation is provided in Simplified Chinese only.

* If you want to update the translation of an existing document, please update messages in ``docs/source/locales``.
* If you have updated a translated English document, we require that the corresponding translated documents to be updated (at least the update should be triggered). Please follow these steps:

  1. Run ``make i18n`` under ``docs`` folder.
  2. Verify that there are new messages in ``docs/source/locales``.
  3. Translate the messages.

* If you intend to translate a new document:

  1. Update ``docs/source/conf.py`` to make ``gettext_documents`` include your document (probably adding a new regular expression).
  2. See the steps above.


To build the translated documentation (for example Chinese documentation), please run:

.. code-block:: bash

   make zh

If you ever encountered problems for translation builds, try to remove the previous build via ``rm -r docs/build/``.

.. _code-of-conduct:

Code of Conduct
---------------

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_.
For more information see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or contact `opencode@microsoft.com <mailto:opencode@microsoft.com>`_ with any additional questions or comments.

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

We enforce every source files in this project to carry a license header. This should be added at the beginning of each file. Please contact the maintainer if you think there should be an exception.

.. tabs::

   .. code-tab:: python

      # Copyright (c) Microsoft Corporation.
      # Licensed under the MIT license.

   .. code-tab:: typescript

      // Copyright (c) Microsoft Corporation.
      // Licensed under the MIT license.
