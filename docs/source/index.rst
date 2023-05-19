NNI Documentation
=================

.. toctree::
   :maxdepth: 2
   :caption: Get Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   hpo/toctree
   nas/toctree
   compression/toctree
   feature_engineering/toctree
   experiment/toctree

.. toctree::
   :maxdepth: 2
   :caption: References
   :hidden:

   Python API <reference/python_api>
   reference/experiment_config
   reference/nnictl

.. toctree::
   :maxdepth: 2
   :caption: Misc
   :hidden:

   examples
   sharings/community_sharings
   notes/research_publications
   notes/build_from_source
   notes/contributing
   release

**NNI (Neural Network Intelligence)** is a lightweight but powerful toolkit to help users **automate**:

* :doc:`Hyperparameter Optimization </hpo/overview>`
* :doc:`Neural Architecture Search </nas/overview>`
* :doc:`Model Compression </compression/overview>`
* :doc:`Feature Engineering </feature_engineering/overview>`

Get Started
-----------

To install the current release:

.. code-block:: bash

   $ pip install nni

See the :doc:`installation guide </installation>` if you need additional help on installation.

Try your first NNI experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   $ nnictl hello

.. note:: You need to have `PyTorch <https://pytorch.org/>`_ (as well as `torchvision <https://pytorch.org/vision/stable/index.html>`_) installed to run this experiment.

To start your journey now, please follow the :doc:`absolute quickstart of NNI <quickstart>`!

Why choose NNI?
---------------

NNI makes AutoML techniques plug-and-play
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <div class="codesnippet-card-container">

.. codesnippetcard::
   :icon: ../img/thumbnails/hpo-small.svg
   :title: Hyperparameter Tuning
   :link: tutorials/hpo_quickstart_pytorch/main

   .. code-block::

      params = nni.get_next_parameter()

      class Net(nn.Module):
          ...

      model = Net()
      optimizer = optim.SGD(model.parameters(),
                            params['lr'],
                            params['momentum'])

      for epoch in range(10):
          train(...)

      accuracy = test(model)
      nni.report_final_result(accuracy)

.. codesnippetcard::
   :icon: ../img/thumbnails/pruning-small.svg
   :title: Model Pruning
   :link: tutorials/pruning_quick_start

   .. code-block::

      # define a config_list
      config = [{
          'sparsity': 0.8,
          'op_types': ['Conv2d']
      }]

      # generate masks for simulated pruning
      wrapped_model, masks = \
          L1NormPruner(model, config). \
          compress()

      # apply the masks for real speedup
      ModelSpeedup(unwrapped_model, input, masks). \
          speedup_model()

.. codesnippetcard::
   :icon: ../img/thumbnails/quantization-small.svg
   :title: Quantization
   :link: tutorials/quantization_quick_start

   .. code-block::

      # define a config_list
      config = [{
          'quant_types': ['input', 'weight'],
          'quant_bits': {'input': 8, 'weight': 8},
          'op_types': ['Conv2d']
      }]

      # in case quantizer needs a extra training
      quantizer = QAT_Quantizer(model, config)
      quantizer.compress()
      # Training...

      # export calibration config and
      # generate TensorRT engine for real speedup
      calibration_config = quantizer.export_model(
          model_path, calibration_path)
      engine = ModelSpeedupTensorRT(
          model, input_shape, config=calib_config)
      engine.compress()

.. codesnippetcard::
   :icon: ../img/thumbnails/multi-trial-nas-small.svg
   :title: Neural Architecture Search
   :link: tutorials/hello_nas

   .. code-block:: python

      # define model space
      class Model(nn.Module):
          self.conv2 = nn.LayerChoice([
              nn.Conv2d(32, 64, 3, 1),
              DepthwiseSeparableConv(32, 64)
          ])
      model_space = Model()
      # search strategy + evaluator
      strategy = RegularizedEvolution()
      evaluator = FunctionalEvaluator(
          train_eval_fn)

      # run experiment
      RetiariiExperiment(model_space,
          evaluator, strategy).run()

.. codesnippetcard::
   :icon: ../img/thumbnails/one-shot-nas-small.svg
   :title: One-shot NAS
   :link: nas/exploration_strategy

   .. code-block::

      # define model space
      space = AnySearchSpace()

      # get a darts trainer
      trainer = DartsTrainer(space, loss, metrics)
      trainer.fit()

      # get final searched architecture
      arch = trainer.export()

.. codesnippetcard::
   :icon: ../img/thumbnails/feature-engineering-small.svg
   :title: Feature Engineering
   :link: feature_engineering/overview

   .. code-block::

      selector = GBDTSelector()
      selector.fit(
          X_train, y_train,
          lgb_params=lgb_params,
          eval_ratio=eval_ratio,
          early_stopping_rounds=10,
          importance_type='gain',
          num_boost_round=1000)

      # get selected features
      features = selector.get_selected_features()

.. End of code snippet card

.. raw:: html

   </div>

NNI eases the effort to scale and manage AutoML experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. codesnippetcard::
   :icon: ../img/thumbnails/training-service-small.svg
   :title: Training Service
   :link: experiment/training_service/overview
   :seemore: See more here.

   An AutoML experiment requires many trials to explore feasible and potentially good-performing models.
   **Training service** aims to make the tuning process easily scalable in a distributed platforms.
   It provides a unified user experience for diverse computation resources (e.g., local machine, remote servers, AKS).
   Currently, NNI supports **more than 9** kinds of training services.

.. codesnippetcard::
   :icon: ../img/thumbnails/web-portal-small.svg
   :title: Web Portal
   :link: experiment/web_portal/web_portal
   :seemore: See more here.

   Web portal visualizes the tuning process, exposing the ability to inspect, monitor and control the experiment.

   .. image:: ../static/img/webui.gif
      :width: 100%

.. codesnippetcard::
   :icon: ../img/thumbnails/experiment-management-small.svg
   :title: Experiment Management
   :link: experiment/experiment_management
   :seemore: See more here.

   The DNN model tuning often requires more than one experiment.
   Users might try different tuning algorithms, fine-tune their search space, or switch to another training service.
   **Experiment management** provides the power to aggregate and compare tuning results from multiple experiments,
   so that the tuning workflow becomes clean and organized.

Get Support and Contribute Back
-------------------------------

NNI is maintained on the `NNI GitHub repository <https://github.com/microsoft/nni>`_. We collect feedbacks and new proposals/ideas on GitHub. You can:

* Open a `GitHub issue <https://github.com/microsoft/nni/issues>`_ for bugs and feature requests.
* Open a `pull request <https://github.com/microsoft/nni/pulls>`_ to contribute code (make sure to read the :doc:`contribution guide <notes/contributing>` before doing this).
* Participate in `NNI Discussion <https://github.com/microsoft/nni/discussions>`_ for general questions and new ideas.
* Join the following IM groups.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Gitter
     - WeChat
   * -
       .. image:: https://user-images.githubusercontent.com/39592018/80665738-e0574a80-8acc-11ea-91bc-0836dc4cbf89.png
     -
       .. image:: https://github.com/scarlett2018/nniutil/raw/master/wechat.png

Citing NNI
----------

If you use NNI in a scientific publication, please consider citing NNI in your references.

   Microsoft. Neural Network Intelligence (version |release|). https://github.com/microsoft/nni

Bibtex entry (please replace the version with the particular version you are using): ::

   @software{nni2021,
      author = {{Microsoft}},
      month = {1},
      title = {{Neural Network Intelligence}},
      url = {https://github.com/microsoft/nni},
      version = {2.0},
      year = {2021}
   }
