.. modified from index.html
.. replace \{\{ pathto\('(.*)'\) \}\} -> $1.html

###########################
Neural Network Intelligence
###########################


..  toctree::
    :maxdepth: 2
    :caption: Get Started
    :hidden:

    Installation <installation>
    QuickStart <Tutorial/QuickStart>
    Tutorials <tutorials>

..  toctree::
    :maxdepth: 2
    :caption: Full-scale Materials
    :hidden:

    Hyperparameter Optimization <hpo/index>
    Neural Architecture Search <nas/index>
    Model Compression <compression/index>
    Feature Engineering <feature_engineering>
    Experiment <experiment/overview>

..  toctree::
    :maxdepth: 2
    :caption: References
    :hidden:

    nnictl Commands <reference/nnictl>
    Experiment Configuration <reference/experiment_config>
    Python API <reference/_modules/nni>

..  toctree::
    :maxdepth: 2
    :caption: Misc
    :hidden:

    Use Cases and Solutions <CommunitySharings/community_sharings>
    Research and Publications <ResearchPublications>
    FAQ <Tutorial/FAQ>
    How to Contribute <contribution>
    Change Log <Release>


.. img:: ../img/nni_logo.png
   :width: 100%
   :align: center

**NNI (Neural Network Intelligence)** is a lightweight but powerful toolkit to help users **automate**:

* `Hyperparameter Tuning </hpo/overview>`_,
* `Neural Architecture Search </nas/index>`_,
* `Model Compression </compression/index>`_,
* `Feature Engineering </FeatureEngineering/Overview>`_.

.. Please keep this part sync with readme

.. admonition:: Latest updates
   :class: attention

   * **New demo available**: `Youtube entry <https://www.youtube.com/channel/UCKcafm6861B2mnYhPbZHavw>`_ | `Bilibili 入口 <https://space.bilibili.com/1649051673>`_ - *last updated on May-26-2021*
   * **New webinar**: `Introducing Retiarii, A deep learning exploratory-training framework on NNI <https://note.microsoft.com/MSR-Webinar-Retiarii-Registration-Live.html>`_ - *scheduled on June-24-2021*
   * **New community channel**: `Discussions <https://github.com/microsoft/nni/discussions>`_
   * **New emoticons release**: `nnSpider <./docs/source/Tutorial/NNSpider.md>`_


.. Can't use section title here due to the limitation of toc

.. rubric:: Install

To install the current release:

```
$ pip install nni
```

To update NNI to the latest version, add `--upgrade` flag to the above commands.

For instructions on building from source, or seeking for help if problems arise when installing from pip, please read the `NNI installation guide </installation>`_.

.. rubric:: NNI makes AutoML techniques plug-and-play.

