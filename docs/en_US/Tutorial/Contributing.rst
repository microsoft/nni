Contributing to Neural Network Intelligence (NNI)
=================================================

Great!! We are always on the lookout for more contributors to our code base.

Firstly, if you are unsure or afraid of anything, just ask or submit the issue or pull request anyways. You won't be yelled at for giving your best effort. The worst that can happen is that you'll be politely asked to change something. We appreciate any sort of contributions and don't want a wall of rules to get in the way of that.

However, for those individuals who want a bit more guidance on the best way to contribute to the project, read on. This document will cover all the points we're looking for in your contributions, raising your chances of quickly merging or addressing your contributions.

Looking for a quickstart, get acquainted with our `Get Started <QuickStart.rst>`__ guide.

There are a few simple guidelines that you need to follow before providing your hacks.

Raising Issues
--------------

When raising issues, please specify the following:


* Setup details needs to be filled as specified in the issue template clearly for the reviewer to check.
* A scenario where the issue occurred (with details on how to reproduce it).
* Errors and log messages that are displayed by the software.
* Any other details that might be useful.

Submit Proposals for New Features
---------------------------------


* 
  There is always something more that is required, to make it easier to suit your use-cases. Feel free to join the discussion on new features or raise a PR with your proposed change.

* 
  Fork the repository under your own github handle. After cloning the repository. Add, commit, push and sqaush (if necessary) the changes with detailed commit messages to your fork. From where you can proceed to making a pull request.

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
  * For docstring to describe ``dict``, which is commonly used in our hyper-param format description, please refer to RiboKit Doc Standards

    * `Internal Guideline on Writing Standards <https://ribokit.github.io/docs/text/>`__

Documentation
-------------

Our documentation is built with :githublink:`sphinx <docs>`.

* Before submitting the documentation change, please **build homepage locally**: ``cd docs/en_US && make html``, then you can see all the built documentation webpage under the folder ``docs/en_US/_build/html``. It's also highly recommended taking care of **every WARNING** during the build, which is very likely the signal of a **deadlink** and other annoying issues.

* 
  For links, please consider using **relative paths** first. However, if the documentation is written in Markdown format, and:


  * It's an image link which needs to be formatted with embedded html grammar, please use global URL like ``https://user-images.githubusercontent.com/44491713/51381727-e3d0f780-1b4f-11e9-96ab-d26b9198ba65.png``, which can be automatically generated by dragging picture onto `Github Issue <https://github.com/Microsoft/nni/issues/new>`__ Box.
  * It cannot be re-formatted by sphinx, such as source code, please use its global URL. For source code that links to our github repo, please use URLs rooted at ``https://github.com/Microsoft/nni/tree/v1.9/`` (:githublink:`mnist.py <examples/trials/mnist-tfv1/mnist.py>` for example).
