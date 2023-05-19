.. role:: raw-html(raw)
   :format: html


Change Log
==========

Release 3.0 Preview - 5/9/2022
------------------------------

Web Portal
^^^^^^^^^^

* New look and feel

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Breaking change**: ``nni.retiarii`` is no longer maintained and tested. Please migrate to ``nni.nas``.

  * Inherit ``nni.nas.nn.pytorch.ModelSpace``, rather than use ``@model_wrapper``.
  * Use ``nni.choice``, rather than ``nni.nas.nn.pytorch.ValueChoice``.
  * Use ``nni.nas.experiment.NasExperiment`` and ``NasExperimentConfig``, rather than ``RetiariiExperiment``.
  * Use ``nni.nas.model_context``, rather than ``nni.nas.fixed_arch``.
  * Please refer to `quickstart <https://nni.readthedocs.io/en/v3.0rc1/tutorials/hello_nas.html>`_ for more changes.

* A refreshed experience to construct model space.
  * Enhanced debuggability via ``freeze()`` and ``simplify()`` APIs.
  * Enhanced expressiveness with ``nni.choice``, ``nni.uniform``, ``nni.normal`` and etc.
  * Enhanced experience of customization with ``MutableModule``, ``ModelSpace`` and ``ParamterizedModule``.
  * Search space with constraints is now supported.

* Improved robustness and stability of strategies.
  * Supported search space types are now enriched for PolicyBaseRL, ENAS and Proxyless.
  * Each step of one-shot strategies can be executed alone: model mutation, evaluator mutation and training.
  * Most multi-trial strategies now supports specifying seed for reproducibility.
  * Performance of strategies have been verified on a set of benchmarks.

* Strategy/engine middleware.
  * Filtering, replicating, deduplicating or retrying models submitted by any strategy.
  * Merging or transforming models before executing (e.g., CGO).
  * Arbitrarily-long chains of middlewares.

* New execution engine.

  * Improved debuggability via SequentialExecutionEngine: trials can run in a single process and breakpoints are effective.
  * The old execution engine is now decomposed into execution engine and model format.
  * Enhanced extensibility of execution engines.

* NAS profiler and hardware-aware NAS.

  * New profilers profile a model space, and quickly compute a profiling result for a sampled architecture or a distribution of architectures (FlopsProfiler, NumParamsProfiler and NnMeterProfiler are officially supported).
  * Assemble profiler with arbitrary strategies, including both multi-trial and one-shot.
  * Profiler are extensible. Strategies can be assembled with arbitrary customized profilers.

Model Compression
^^^^^^^^^^^^^^^^^

* Compression framework is refactored, new framework import path is ``nni.contrib.compression``.

  * Configure keys are refactored, support more detailed compression configurations.
  * Support multi compression methods fusion.
  * Support distillation as a basic compression component.
  * Support more compression targets, like ``input``, ``ouptut`` and any registered paramters.
  * Support compressing any module type by customizing module settings.

* Pruning

  * Pruner interfaces have fine-tuned for easy to use.
  * Support configuring ``granularity`` in pruners.
  * Support different mask ways, multiply zero or add a large negative value.
  * Support manully setting dependency group and global group.
  * A new powerful pruning speedup is released, applicability and robustness have been greatly improved.
  * The end to end transformer compression tutorial has been updated, achieved more extreme compression performance.

* Quantization

  * Support using ``Evaluator`` to handle training/inferencing.
  * Support more module fusion combinations.
  * Support configuring ``granularity`` in quantizers.

* Distillation

  * DynamicLayerwiseDistiller and Adaptive1dLayerwiseDistiller are supported.

* Compression documents now updated for the new framework, the old version please view `v2.10 <https://nni.readthedocs.io/en/v2.10/>`_ doc.
* New compression examples are under `nni/examples/compression <https://github.com/microsoft/nni/tree/v3.0rc1/examples/compression>`_

  * Create a evaluator: `nni/examples/compression/evaluator <https://github.com/microsoft/nni/tree/v3.0rc1/examples/compression/evaluator>`_
  * Pruning a model: `nni/examples/compression/pruning <https://github.com/microsoft/nni/tree/v3.0rc1/examples/compression/pruning>`_
  * Quantize a model: `nni/examples/compression/quantization <https://github.com/microsoft/nni/tree/v3.0rc1/examples/compression/quantization>`_
  * Fusion compression: `nni/examples/compression/fusion <https://github.com/microsoft/nni/tree/v3.0rc1/examples/compression/fusion>`_

Training Services
^^^^^^^^^^^^^^^^^

* **Breaking change**: NNI v3.0 cannot resume experiments created by NNI v2.x
* Local training service:

  * Reduced latency of creating trials
  * Fixed "GPU metric not found"
  * Fixed bugs about resuming trials

* Remote training service:

  * ``reuse_mode`` now defaults to ``False``; setting it to ``True`` will fallback to v2.x remote training service
  * Reduced latency of creating trials
  * Fixed "GPU metric not found"
  * Fixed bugs about resuming trials
  * Supported viewing trial logs on the web portal
  * Supported automatic recover after temporary server failure (network fluctuation, out of memory, etc)

Release 2.10 - 11/14/2022
-------------------------

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

*  Added trial deduplication for evolutionary search.
*  Fixed the racing issue in RL strategy on submitting models.
*  Fixed an issue introduced by the trial recovery feature.
*  Fixed import error of ``PyTorch Lightning`` in NAS.

Compression
^^^^^^^^^^^

*  Supported parsing schema by replacing ``torch._C.parse_schema`` in pytorch 1.8.0 in ModelSpeedup.
*  Fixed the bug that speedup ``rand_like_with_shape`` is easy to overflow when ``dtype=torch.int8``.
*  Fixed the propagation error with view tensors in speedup.

Hyper-parameter optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  Supported rerunning the interrupted trials induced by the termination of an NNI experiment when resuming this experiment.
*  Fixed a dependency issue of Anneal tuner by changing Anneal tuner dependency to optional.
*  Fixed a bug that tuner might lose connection in long experiments.

Training service
^^^^^^^^^^^^^^^^

*  Fixed a bug that trial code directory cannot have non-English characters.

Web portal
^^^^^^^^^^

*  Fixed an error of columns in HPO experiment hyper-parameters page by using localStorage.
*  Fixed a link error in About menu on WebUI.

Known issues
^^^^^^^^^^^^

*  Modelspeedup does not support non-tensor intermediate variables.

Release 2.9 - 9/8/2022
----------------------

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

*  New tutorial of model space hub and one-shot strategy.
   (`tutorial <https://nni.readthedocs.io/en/v2.9/tutorials/darts.html>`__)
*  Add pretrained checkpoints to AutoFormer.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/nas/search_space.htm.retiarii.hub.pytorch.AutoformerSpace>`__)
*  Support loading checkpoint of a trained supernet in a subnet.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/nas/strategy.htm.retiarii.strategy.RandomOneShot>`__)
*  Support view and resume of NAS experiment.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/nas/others.htm.retiarii.experiment.pytorch.RetiariiExperiment.resume>`__)

Enhancements
""""""""""""

*  Support ``fit_kwargs`` in lightning evaluator.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/nas/evaluator.html#nni.retiarii.evaluator.pytorch.Lightning>`__)
*  Support ``drop_path`` and ``auxiliary_loss`` in NASNet.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/nas/search_space.html#nasnet>`__)
*  Support gradient clipping in DARTS.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/nas/strategy.html#nni.retiarii.strategy.DARTS>`__)
*  Add ``export_probs`` to monitor the architecture weights.
*  Rewrite configure_optimizers, functions to step optimizers /
   schedulers, along with other hooks for simplicity, and to be
   compatible with latest lightning (v1.7).
*  Align implementation of DifferentiableCell with DARTS official repo.
*  Re-implementation of ProxylessNAS.
*  Move ``nni.retiarii`` code-base to ``nni.nas``.

Bug fixes
"""""""""

*  Fix a performance issue caused by tensor formatting in ``weighted_sum``.
*  Fix a misuse of lambda expression in NAS-Bench-201 search space.
*  Fix the gumbel temperature schedule in Gumbel DARTS.
*  Fix the architecture weight sharing when sharing labels in differentiable strategies.
*  Fix the memo reusing in exporting differentiable cell.

Compression
^^^^^^^^^^^

*  New tutorial of pruning transformer model.
   (`tutorial <https://nni.readthedocs.io/en/v2.9/tutorials/pruning_bert_glue.html>`__)
*  Add ``TorchEvaluator``, ``LightningEvaluator``, ``TransformersEvaluator``
   to ease the expression of training logic in pruner.
   (`doc <https://nni.readthedocs.io/en/v2.9/compression/compression_evaluator.html>`__,
   `API <https://nni.readthedocs.io/en/v2.9/reference/compression/evaluator.html>`__)

Enhancements
""""""""""""

*  Promote all pruner API using ``Evaluator``, the old API is deprecated and will be removed in v3.0.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/compression/pruner.html>`__)
*  Greatly enlarge the set of supported operators in pruning speedup via automatic operator conversion.
*  Support ``lr_scheduler`` in pruning by using ``Evaluator``.
*  Support pruning NLP task in ``ActivationAPoZRankPruner`` and ``ActivationMeanRankPruner``.
*  Add ``training_steps``, ``regular_scale``, ``movement_mode``, ``sparse_granularity`` for ``MovementPruner``.
   (`doc <https://nni.readthedocs.io/en/v2.9/reference/compression/pruner.html#movement-pruner>`__)
*  Add ``GroupNorm`` replacement in pruning speedup. Thanks external contributor
   `@cin-xing <https://github.com/cin-xing>`__.
*  Optimize ``balance`` mode performance in ``LevelPruner``.

Bug fixes
"""""""""

*  Fix the invalid ``dependency_aware`` mode in scheduled pruners.
*  Fix the bug where ``bias`` mask cannot be generated.
*  Fix the bug where ``max_sparsity_per_layer`` has no effect.
*  Fix ``Linear`` and ``LayerNorm`` speedup replacement in NLP task.
*  Fix tracing ``LightningModule`` failed in ``pytorch_lightning >= 1.7.0``.

Hyper-parameter optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  Fix the bug that weights are not defined correctly in ``adaptive_parzen_normal`` of TPE.

Training service
^^^^^^^^^^^^^^^^

*  Fix trialConcurrency bug in K8S training service: use``${envId}_run.sh`` to replace ``run.sh``.
*  Fix upload dir bug in K8S training service: use a separate working
   directory for each experiment. Thanks external contributor
   `@amznero <https://github.com/amznero>`__.

Web portal
^^^^^^^^^^

*  Support dict keys in Default metric chart in the detail page.
*  Show experiment error message with small popup windows in the bottom right of the page.
*  Upgrade React router to v6 to fix index router issue.
*  Fix the issue of details page crashing due to choices containing ``None``.
*  Fix the issue of missing dict intermediate dropdown in comparing trials dialog.

Known issues
^^^^^^^^^^^^

*  Activation based pruner can not support ``[batch, seq, hidden]``.
*  Failed trials are NOT auto-submitted when experiment is resumed
   (`[FEAT]: resume waiting/running, dedup on tuner side
   (TPE-only) #4931 <https://github.com/microsoft/nni/pull/4931>`__ is
   reverted due to its pitfalls).

Release 2.8 - 6/22/2022
-----------------------

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Align user experience of one-shot NAS with multi-trial NAS, i.e., users can use one-shot NAS by specifying the corresponding strategy (`doc <https://nni.readthedocs.io/en/v2.8/nas/exploration_strategy.html#one-shot-strategy>`__)
* Support multi-GPU training of one-shot NAS
* *Preview* Support load/retrain the pre-searched model of some search spaces, i.e., 18 models in 4 different search spaces (`doc <https://github.com/microsoft/nni/tree/v2.8/nni/retiarii/hub>`__)
* Support AutoFormer search space in search space hub, thanks our collaborators @nbl97 and @penghouwen
* One-shot NAS supports the NAS API ``repeat`` and ``cell``
* Refactor of RetiariiExperiment to share the common implementation with HPO experiment
* CGO supports pytorch-lightning 1.6

Model Compression
^^^^^^^^^^^^^^^^^

* *Preview* Refactor and improvement of automatic model compress with a new ``CompressionExperiment``
* Support customizating module replacement function for unsupported modules in model speedup (`doc <https://nni.readthedocs.io/en/v2.8/reference/compression/pruning_speedup.html#nni.compression.pytorch.speedup.ModelSpeedup>`__)
* Support the module replacement function for some user mentioned modules
* Support output_padding for convtranspose2d in model speedup, thanks external contributor @haoshuai-orka

Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Make ``config.tuner.name`` case insensitive
* Allow writing configurations of advisor in tuner format, i.e., aligning the configuration of advisor and tuner

Experiment
^^^^^^^^^^

* Support launching multiple HPO experiments in one process
* Internal refactors and improvements

  * Refactor of the logging mechanism in NNI
  * Refactor of NNI manager globals for flexible and high extensibility
  * Migrate dispatcher IPC to WebSocket
  * Decouple lock stuffs from experiments manager logic
  * Use launcher's sys.executable to detect Python interpreter

WebUI
^^^^^

* Improve user experience of trial ordering in the overview page
* Fix the update issue in the trial detail page

Documentation
^^^^^^^^^^^^^

* A new translation framework for document
* Add a new quantization demo (`doc <https://nni.readthedocs.io/en/v2.8/tutorials/quantization_quick_start_mnist.html>`__)

Notable Bugfixes
^^^^^^^^^^^^^^^^

* Fix TPE import issue for old metrics
* Fix the issue in TPE nested search space
* Support ``RecursiveScriptModule`` in speedup
* Fix the issue of failed "implicit type cast" in merge_parameter()

Release 2.7 - 4/18/2022
-----------------------

Documentation
^^^^^^^^^^^^^

A full-size upgrade of the documentation, with the following significant improvements in the reading experience, practical tutorials, and examples:

* Reorganized the document structure with a new document template. (`Upgraded doc entry <https://nni.readthedocs.io/en/v2.7>`__)
* Add more friendly tutorials with jupyter notebook. (`New Quick Starts <https://nni.readthedocs.io/en/v2.7/quickstart.html>`__)
* New model pruning demo available. (`Youtube entry <https://www.youtube.com/channel/UCKcafm6861B2mnYhPbZHavw>`__, `Bilibili entry <https://space.bilibili.com/1649051673>`__)

Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [Improvement] TPE and random tuners will not generate duplicate hyperparameters anymore.
* [Improvement] Most Python APIs now have type annotations.

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Jointly search for architecture and hyper-parameters: ValueChoice in evaluator. (`doc <https://nni.readthedocs.io/en/v2.7/reference/nas/search_space.html#valuechoice>`__)
* Support composition (transformation) of one or several value choices. (`doc <https://nni.readthedocs.io/en/v2.7/reference/nas/search_space.html#valuechoice>`__)
* Enhanced Cell API (``merge_op``, preprocessor, postprocessor). (`doc <https://nni.readthedocs.io/en/v2.7/reference/nas/search_space.html#cell>`__)
* The argument ``depth`` in the ``Repeat`` API allows ValueChoice. (`doc <https://nni.readthedocs.io/en/v2.7/reference/nas/search_space.html#repeat>`__)
* Support loading ``state_dict`` between sub-net and super-net. (`doc <https://nni.readthedocs.io/en/v2.7/reference/nas/others.html#nni.retiarii.utils.original_state_dict_hooks>`__, `example in spos <https://nni.readthedocs.io/en/v2.7/reference/nas/strategy.html#spos>`__)
* Support BN fine-tuning and evaluation in SPOS example. (`doc <https://nni.readthedocs.io/en/v2.7/reference/nas/strategy.html#spos>`__)
* *Experimental* Model hyper-parameter choice. (`doc <https://nni.readthedocs.io/en/v2.7/reference/nas/search_space.html#modelparameterchoice>`__)
* *Preview* Lightning implementation for Retiarii including DARTS, ENAS, ProxylessNAS and RandomNAS. (`example usage <https://github.com/microsoft/nni/blob/v2.7/test/ut/retiarii/test_oneshot.py>`__)
* *Preview* A search space hub that contains 10 search spaces. (`code <https://github.com/microsoft/nni/tree/v2.7/nni/retiarii/hub>`__)

Model Compression
^^^^^^^^^^^^^^^^^

* Pruning V2 is promoted as default pruning framework, old pruning is legacy and keeps for a few releases.(`doc <https://nni.readthedocs.io/en/v2.7/reference/compression/pruner.html>`__)
* A new pruning mode ``balance`` is supported in ``LevelPruner``.(`doc <https://nni.readthedocs.io/en/v2.7/reference/compression/pruner.html#level-pruner>`__)
* Support coarse-grained pruning in ``ADMMPruner``.(`doc <https://nni.readthedocs.io/en/v2.7/reference/compression/pruner.html#admm-pruner>`__)
* [Improvement] Support more operation types in pruning speedup.
* [Improvement] Optimize performance of some pruners.

Experiment
^^^^^^^^^^

* [Improvement] Experiment.run() no longer stops web portal on return.

Notable Bugfixes
^^^^^^^^^^^^^^^^

* Fixed: experiment list could not open experiment with prefix.
* Fixed: serializer for complex kinds of arguments.
* Fixed: some typos in code. (thanks @a1trl9 @mrshu)
* Fixed: dependency issue across layer in pruning speedup. 
* Fixed: uncheck trial doesn't work bug in the detail table.
* Fixed: filter name | id bug in the experiment management page.

Release 2.6 - 1/19/2022
-----------------------

**NOTE**: NNI v2.6 is the last version that supports Python 3.6. From next release NNI will require Python 3.7+.

Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experiment
""""""""""

* The legacy experiment config format is now deprecated. `(doc of new config) <https://nni.readthedocs.io/en/v2.6/reference/experiment_config.html>`__

  * If you are still using legacy format, nnictl will show equivalent new config on start. Please save it to replace the old one.

* nnictl now uses ``nni.experiment.Experiment`` `APIs <https://nni.readthedocs.io/en/stable/Tutorial/HowToLaunchFromPython.html>`__ as backend. The output message of create, resume, and view commands have changed.
* Added Kubeflow and Frameworkcontroller support to hybrid mode.  `(doc) <https://nni.readthedocs.io/en/v2.6/TrainingService/HybridMode.html>`__
* The hidden tuner manifest file has been updated. This should be transparent to users, but if you encounter issues like failed to find tuner, please try to remove ``~/.config/nni``.

Algorithms
""""""""""

* Random tuner now supports classArgs ``seed``. `(doc) <https://nni.readthedocs.io/en/v2.6/Tuner/RandomTuner.html>`__
* TPE tuner is refactored: `(doc) <https://nni.readthedocs.io/en/v2.6/Tuner/TpeTuner.html>`__

  * Support classArgs ``seed``.
  * Support classArgs ``tpe_args`` for expert users to customize algorithm behavior.
  * Parallel optimization has been turned on by default. To turn it off set ``tpe_args.constant_liar_type`` to ``null`` (or ``None`` in Python).
  * ``parallel_optimize`` and ``constant_liar_type`` has been removed. If you are using them please update your config to use ``tpe_args.constant_liar_type`` instead.

* Grid search tuner now supports all search space types, including uniform, normal, and nested choice. `(doc) <https://nni.readthedocs.io/en/v2.6/Tuner/GridsearchTuner.html>`__

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Enhancement to serialization utilities `(doc) <https://nni.readthedocs.io/en/v2.6/NAS/Serialization.html>`__ and changes to recommended practice of customizing evaluators. `(doc) <https://nni.readthedocs.io/en/v2.6/NAS/QuickStart.html#pick-or-customize-a-model-evaluator>`__
* Support latency constraint on edge device for ProxylessNAS based on nn-Meter. `(doc) <https://nni.readthedocs.io/en/v2.6/NAS/Proxylessnas.html>`__
* Trial parameters are showed more friendly in Retiarii experiments.
* Refactor NAS examples of ProxylessNAS and SPOS.

Model Compression
^^^^^^^^^^^^^^^^^

* New Pruner Supported in Pruning V2

  * Auto-Compress Pruner `(doc) <https://nni.readthedocs.io/en/v2.6/Compression/v2_pruning_algo.html#auto-compress-pruner>`__
  * AMC Pruner `(doc) <https://nni.readthedocs.io/en/v2.6/Compression/v2_pruning_algo.html#amc-pruner>`__
  * Movement Pruning Pruner `(doc) <https://nni.readthedocs.io/en/v2.6/Compression/v2_pruning_algo.html#movement-pruner>`__

* Support ``nni.trace`` wrapped ``Optimizer`` in Pruning V2. In the case of not affecting the user experience as much as possible, trace the input parameters of the optimizer. `(doc) <https://nni.readthedocs.io/en/v2.6/Compression/v2_pruning_algo.html>`__
* Optimize Taylor Pruner, APoZ Activation Pruner, Mean Activation Pruner in V2 memory usage.
* Add more examples for Pruning V2.
* Add document for pruning config list.  `(doc) <https://nni.readthedocs.io/en/v2.6/Compression/v2_pruning_config_list.html>`__
* Parameter ``masks_file`` of ``ModelSpeedup`` now accepts `pathlib.Path` object. (Thanks to @dosemeion) `(doc) <https://nni.readthedocs.io/en/v2.6/Compression/ModelSpeedup.html#user-configuration-for-modelspeedup>`__
* Bug Fix

  * Fix Slim Pruner in V2 not sparsify the BN weight.
  * Fix Simulator Annealing Task Generator generates config ignoring 0 sparsity.

Documentation
^^^^^^^^^^^^^

* Supported GitHub feature "Cite this repository".
* Updated index page of readthedocs.
* Updated Chinese documentation.

  * From now on NNI only maintains translation for most import docs and ensures they are up to date.

* Reorganized HPO tuners' doc.

Bugfixes
^^^^^^^^

* Fixed a bug where numpy array is used as a truth value. (Thanks to @khituras)
* Fixed a bug in updating search space.
* Fixed a bug that HPO search space file does not support scientific notation and tab indent.

  * For now NNI does not support mixing scientific notation and YAML features. We are waiting for PyYAML to update.

* Fixed a bug that causes DARTS 2nd order to crash.
* Fixed a bug that causes deep copy of mutation primitives (e.g., LayerChoice) to crash.
* Removed blank at bottom in Web UI overview page.

Release 2.5 - 11/2/2021
-----------------------

Model Compression
^^^^^^^^^^^^^^^^^

* New major version of pruning framework `(doc) <https://nni.readthedocs.io/en/v2.5/Compression/v2_pruning.html>`__

  * Iterative pruning is more automated, users can use less code to implement iterative pruning.
  * Support exporting intermediate models in the iterative pruning process.
  * The implementation of the pruning algorithm is closer to the paper.
  * Users can easily customize their own iterative pruning by using ``PruningScheduler``.
  * Optimize the basic pruners underlying generate mask logic, easier to extend new functions.
  * Optimized the memory usage of the pruners.

* MobileNetV2 end-to-end example `(notebook) <https://github.com/microsoft/nni/blob/v2.5/examples/model_compress/pruning/mobilenetv2_end2end/Compressing%20MobileNetV2%20with%20NNI%20Pruners.ipynb>`__
* Improved QAT quantizer `(doc) <https://nni.readthedocs.io/en/v2.5/Compression/Quantizer.html#qat-quantizer>`__

  * support dtype and scheme customization
  * support dp multi-gpu training
  * support load_calibration_config

* Model speed-up now supports directly loading the mask `(doc) <https://nni.readthedocs.io/en/v2.5/Compression/ModelSpeedup.html#nni.compression.pytorch.ModelSpeedup>`__
* Support speed-up depth-wise convolution
* Support bn-folding for LSQ quantizer
* Support QAT and LSQ resume from PTQ
* Added doc for observer quantizer `(doc) <https://nni.readthedocs.io/en/v2.5/Compression/Quantizer.html#observer-quantizer>`__

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^

* NAS benchmark `(doc) <https://nni.readthedocs.io/en/v2.5/NAS/Benchmarks.html>`__

  * Support benchmark table lookup in experiments
  * New data preparation approach

* Improved `quick start doc <https://nni.readthedocs.io/en/v2.5/NAS/QuickStart.html>`__
* Experimental CGO execution engine `(doc) <https://nni.readthedocs.io/en/v2.5/NAS/ExecutionEngines.html#cgo-execution-engine-experimental>`__

Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* New training platform: Alibaba DSW+DLC `(doc) <https://nni.readthedocs.io/en/v2.5/TrainingService/DLCMode.html>`__
* Support passing ConfigSpace definition directly to BOHB `(doc) <https://nni.readthedocs.io/en/v2.5/Tuner/BohbAdvisor.html#usage>`__ (thanks to khituras)
* Reformatted `experiment config doc <https://nni.readthedocs.io/en/v2.5/reference/experiment_config.html>`__
* Added example config files for Windows (thanks to @politecat314)
* FrameworkController now supports reuse mode

Fixed Bugs
^^^^^^^^^^

* Experiment cannot start due to platform timestamp format (issue #4077 #4083)
* Cannot use ``1e-5`` in search space (issue #4080)
* Dependency version conflict caused by ConfigSpace (issue #3909) (thanks to @jexxers)
* Hardware-aware SPOS example does not work (issue #4198)
* Web UI show wrong remaining time when duration exceeds limit (issue #4015)
* cudnn.deterministic is always set in AMC pruner (#4117) thanks to @mstczuo

And...
^^^^^^

* New `emoticons <https://github.com/microsoft/nni/blob/v2.5/docs/en_US/Tutorial/NNSpider.md>`__!

.. image:: https://raw.githubusercontent.com/microsoft/nni/v2.5/docs/img/emoicons/Holiday.png

Release 2.4 - 8/11/2021
-----------------------

Major Updates
^^^^^^^^^^^^^

Neural Architecture Search
""""""""""""""""""""""""""

* NAS visualization: visualize model graph through Netron (#3878)
* Support NAS bench 101/201 on Retiarii framework (#3871 #3920)
* Support hypermodule AutoActivation (#3868)
* Support PyTorch v1.8/v1.9 (#3937)
* Support Hardware-aware NAS with nn-Meter (#3938)
* Enable `fixed_arch` on Retiarii (#3972)

Model Compression
"""""""""""""""""

* Refactor of ModelSpeedup: auto shape/mask inference (#3462)
* Added more examples for ModelSpeedup (#3880)
* Support global sort for Taylor pruning (#3896)
* Support TransformerHeadPruner (#3884)
* Support batch normalization folding in QAT quantizer (#3911, thanks the external contributor @chenbohua3)
* Support post-training observer quantizer (#3915, thanks the external contributor @chenbohua3)
* Support ModelSpeedup for Slim Pruner (#4008)
* Support TensorRT 8.0.0 in ModelSpeedup (#3866)

Hyper-parameter Tuning
""""""""""""""""""""""

* Improve HPO benchmarks (#3925)
* Improve type validation of user defined search space (#3975)

Training service & nnictl
"""""""""""""""""""""""""

* Support JupyterLab (#3668 #3954)
* Support viewing experiment from experiment folder (#3870)
* Support kubeflow in training service reuse framework (#3919)
* Support viewing trial log on WebUI for an experiment launched in `view` mode (#3872)

Minor Updates & Bug Fixes
"""""""""""""""""""""""""

* Fix the failure of the exit of Retiarii experiment (#3899)
* Fix `exclude` not supported in some `config_list` cases (#3815)
* Fix bug in remote training service on reuse mode (#3941)
* Improve IP address detection in modern way (#3860)
* Fix bug of the search box on WebUI (#3935)
* Fix bug in url_prefix of WebUI (#4051)
* Support dict format of intermediate on WebUI (#3895)
* Fix bug in openpai training service induced by experiment config v2 (#4027 #4057)
* Improved doc (#3861 #3885 #3966 #4004 #3955)
* Improved the API `export_model` in model compression (#3968)
* Supported `UnSqueeze` in ModelSpeedup (#3960)
* Thanks other external contributors: @Markus92 (#3936), @thomasschmied (#3963), @twmht (#3842)


Release 2.3 - 6/15/2021
-----------------------

Major Updates
^^^^^^^^^^^^^

Neural Architecture Search
""""""""""""""""""""""""""

* Retiarii Framework (NNI NAS 2.0) Beta Release with new features:

  * Support new high-level APIs: ``Repeat`` and ``Cell`` (#3481)
  * Support pure-python execution engine (#3605)
  * Support policy-based RL strategy (#3650)
  * Support nested ModuleList (#3652)
  * Improve documentation (#3785)

  **Note**: there are more exciting features of Retiarii planned in the future releases, please refer to `Retiarii Roadmap <https://github.com/microsoft/nni/discussions/3744>`__  for more information.

* Add new NAS algorithm: Blockwise DNAS FBNet (#3532, thanks the external contributor @alibaba-yiwuyao) 

Model Compression
"""""""""""""""""

* Support Auto Compression Framework (#3631)
* Support slim pruner in Tensorflow (#3614)
* Support LSQ quantizer (#3503, thanks the external contributor @chenbohua3)
* Improve APIs for iterative pruners (#3507 #3688)

Training service & Rest
"""""""""""""""""""""""

* Support 3rd-party training service (#3662 #3726)
* Support setting prefix URL (#3625 #3674 #3672 #3643)
* Improve NNI manager logging (#3624)
* Remove outdated TensorBoard code on nnictl (#3613)

Hyper-Parameter Optimization
""""""""""""""""""""""""""""

* Add new tuner: DNGO (#3479 #3707)
* Add benchmark for tuners (#3644 #3720 #3689)

WebUI
"""""

* Improve search parameters on trial detail page (#3651 #3723 #3715)
* Make selected trials consistent after auto-refresh in detail table (#3597)
* Add trial stdout button on local mode (#3653 #3690)

Examples & Documentation
""""""""""""""""""""""""

* Convert all trial examples' from config v1 to config v2 (#3721 #3733 #3711 #3600)
* Add new jupyter notebook examples (#3599 #3700)

Dev Excellent
"""""""""""""

* Upgrade dependencies in Dockerfile (#3713 #3722)
* Substitute PyYAML for ``ruamel.yaml`` (#3702)
* Add pipelines for AML and hybrid training service and experiment config V2 (#3477 #3648)
* Add pipeline badge in README (#3589)
* Update issue bug report template (#3501)


Bug Fixes & Minor Updates
^^^^^^^^^^^^^^^^^^^^^^^^^

* Fix syntax error on Windows (#3634)
* Fix a logging related bug (#3705)
* Fix a bug in GPU indices (#3721)
* Fix a bug in FrameworkController (#3730)
* Fix a bug in ``export_data_url format`` (#3665)
* Report version check failure as a warning (#3654)
* Fix bugs and lints in nnictl (#3712)
* Fix bug of ``optimize_mode`` on WebUI (#3731)
* Fix bug of ``useActiveGpu`` in AML v2 config (#3655)
* Fix bug of ``experiment_working_directory`` in Retiarii config (#3607)
* Fix a bug in mask conflict (#3629, thanks the external contributor @Davidxswang) 
* Fix a bug in model speedup shape inference (#3588, thanks the external contributor @Davidxswang)
* Fix a bug in multithread on Windows (#3604, thanks the external contributor @Ivanfangsc)
* Delete redundant code in training service (#3526, thanks the external contributor @maxsuren)
* Fix typo in DoReFa compression doc (#3693, thanks the external contributor @Erfandarzi)
* Update docstring in model compression (#3647, thanks the external contributor @ichejun)
* Fix a bug when using Kubernetes container (#3719, thanks the external contributor @rmfan)


Release 2.2 - 4/26/2021
-----------------------

Major updates
^^^^^^^^^^^^^

Neural Architecture Search
""""""""""""""""""""""""""

* Improve NAS 2.0 (Retiarii) Framework (Alpha Release)

  * Support local debug mode (#3476)
  * Support nesting ``ValueChoice`` in ``LayerChoice`` (#3508)
  * Support dict/list type in ``ValueChoice`` (#3508)
  * Improve the format of export architectures (#3464)
  * Refactor of NAS examples (#3513)
  * Refer to `here <https://github.com/microsoft/nni/issues/3301>`__ for Retiarii Roadmap

Model Compression
"""""""""""""""""

* Support speedup for mixed precision quantization model (Experimental) (#3488 #3512)
* Support model export for quantization algorithm (#3458 #3473)
* Support model export in model compression for TensorFlow (#3487)
* Improve documentation (#3482)

nnictl & nni.experiment
"""""""""""""""""""""""

* Add native support for experiment config V2 (#3466 #3540 #3552)
* Add resume and view mode in Python API ``nni.experiment`` (#3490 #3524 #3545)

Training Service
""""""""""""""""

* Support umount for shared storage in remote training service (#3456)
* Support Windows as the remote training service in reuse mode (#3500)
* Remove duplicated env folder in remote training service (#3472)
* Add log information for GPU metric collector (#3506)
* Enable optional Pod Spec for FrameworkController platform (#3379, thanks the external contributor @mbu93)

WebUI
"""""

* Support launching TensorBoard on WebUI (#3454 #3361 #3531)
* Upgrade echarts-for-react to v5 (#3457)
* Add wrap for dispatcher/nnimanager log monaco editor (#3461)

Bug Fixes
^^^^^^^^^

* Fix bug of FLOPs counter (#3497)
* Fix bug of hyper-parameter Add/Remove axes and table Add/Remove columns button conflict (#3491)
* Fix bug that monaco editor search text is not displayed completely (#3492)
* Fix bug of Cream NAS (#3498, thanks the external contributor @AliCloud-PAI)
* Fix typos in docs (#3448, thanks the external contributor @OliverShang)
* Fix typo in NAS 1.0 (#3538, thanks the external contributor @ankitaggarwal23)


Release 2.1 - 3/10/2021
-----------------------

Major updates
^^^^^^^^^^^^^

Neural architecture search
""""""""""""""""""""""""""

* Improve NAS 2.0 (Retiarii) Framework (Improved Experimental)

  * Improve the robustness of graph generation and code generation for PyTorch models (#3365)
  * Support the inline mutation API ``ValueChoice`` (#3349 #3382)
  * Improve the design and implementation of Model Evaluator (#3359 #3404)
  * Support Random/Grid/Evolution exploration strategies (i.e., search algorithms) (#3377)
  * Refer to `here <https://github.com/microsoft/nni/issues/3301>`__ for Retiarii Roadmap

Training service
""""""""""""""""

* Support shared storage for reuse mode (#3354)
* Support Windows as the local training service in hybrid mode (#3353)
* Remove PAIYarn training service (#3327)
* Add "recently-idle" scheduling algorithm (#3375)
* Deprecate ``preCommand`` and enable ``pythonPath`` for remote training service (#3284 #3410)
* Refactor reuse mode temp folder (#3374)

nnictl & nni.experiment
"""""""""""""""""""""""

* Migrate ``nnicli`` to new Python API ``nni.experiment`` (#3334)
* Refactor the way of specifying tuner in experiment Python API (\ ``nni.experiment``\ ), more aligned with ``nnictl`` (#3419)

WebUI
"""""

* Support showing the assigned training service of each trial in hybrid mode on WebUI (#3261 #3391)
* Support multiple selection for filter status in experiments management page (#3351)
* Improve overview page (#3316 #3317 #3352)
* Support copy trial id in the table (#3378)

Documentation
^^^^^^^^^^^^^

* Improve model compression examples and documentation (#3326 #3371)
* Add Python API examples and documentation (#3396)
* Add SECURITY doc (#3358)
* Add 'What's NEW!' section in README (#3395) 
* Update English contributing doc (#3398, thanks external contributor @Yongxuanzhang)

Bug fixes
^^^^^^^^^

* Fix AML outputs path and python process not killed (#3321)
* Fix bug that an experiment launched from Python cannot be resumed by nnictl (#3309)
* Fix import path of network morphism example (#3333)
* Fix bug in the tuple unpack (#3340)
* Fix bug of security for arbitrary code execution (#3311, thanks external contributor @huntr-helper)
* Fix ``NoneType`` error on jupyter notebook (#3337, thanks external contributor @tczhangzhi)
* Fix bugs in Retiarii (#3339 #3341 #3357, thanks external contributor @tczhangzhi)
* Fix bug in AdaptDL mode example (#3381, thanks external contributor @ZeyaWang)
* Fix the spelling mistake of assessor (#3416, thanks external contributor @ByronCHAO)
* Fix bug in ruamel import (#3430, thanks external contributor @rushtehrani)


Release 2.0 - 1/14/2021
-----------------------

Major updates
^^^^^^^^^^^^^

Neural architecture search
""""""""""""""""""""""""""

* Support an improved NAS framework: Retiarii (experimental)

  * Feature roadmap (`issue #3301 <https://github.com/microsoft/nni/issues/3301>`__)
  * `Related issues and pull requests <https://github.com/microsoft/nni/issues?q=label%3Aretiarii-v2.0>`__
  * Documentation (#3221 #3282 #3287)

* Support a new NAS algorithm: Cream (#2705)
* Add a new NAS benchmark for NLP model search (#3140)

Training service
""""""""""""""""

* Support hybrid training service (#3097 #3251 #3252)
* Support AdlTrainingService, a new training service based on Kubernetes (#3022, thanks external contributors Petuum @pw2393)


Model compression
"""""""""""""""""

* Support pruning schedule for fpgm pruning algorithm (#3110)
* ModelSpeedup improvement: support torch v1.7 (updated graph_utils.py) (#3076)
* Improve model compression utility: model flops counter (#3048 #3265)


WebUI & nnictl 
""""""""""""""

* Support experiments management on WebUI, add a web page for it (#3081 #3127)
* Improve the layout of overview page (#3046 #3123)
* Add navigation bar on the right for logs and configs; add expanded icons for table (#3069 #3103)


Others
""""""

* Support launching an experiment from Python code (#3111 #3210 #3263)
* Refactor builtin/customized tuner installation (#3134)
* Support new experiment configuration V2 (#3138 #3248 #3251)
* Reorganize source code directory hierarchy (#2962 #2987 #3037)
* Change SIGKILL to SIGTERM in local mode when cancelling trial jobs (#3173)
* Refector hyperband (#3040)


Documentation
^^^^^^^^^^^^^

* Port markdown docs to reStructuredText docs and introduce ``githublink`` (#3107)
* List related research and publications in doc (#3150)
* Add tutorial of saving and loading quantized model (#3192)
* Remove paiYarn doc and add description of ``reuse`` config in remote mode (#3253)
* Update EfficientNet doc to clarify repo versions (#3158, thanks external contributor @ahundt)

Bug fixes
^^^^^^^^^

* Fix exp-duration pause timing under NO_MORE_TRIAL status (#3043)
* Fix bug in NAS SPOS trainer, apply_fixed_architecture (#3051, thanks external contributor @HeekangPark)
* Fix ``_compute_hessian`` bug in NAS DARTS (PyTorch version) (#3058, thanks external contributor @hroken)
* Fix bug of conv1d in the cdarts utils (#3073, thanks external contributor @athaker)
* Fix the handling of unknown trials when resuming an experiment (#3096)
* Fix bug of kill command under Windows (#3106)
* Fix lazy logging (#3108, thanks external contributor @HarshCasper)
* Fix checkpoint load and save issue in QAT quantizer (#3124, thanks external contributor @eedalong)
* Fix quant grad function calculation error (#3160, thanks external contributor @eedalong)
* Fix device assignment bug in quantization algorithm (#3212, thanks external contributor @eedalong)
* Fix bug in ModelSpeedup and enhance UT for it (#3279)
* and others (#3063 #3065 #3098 #3109 #3125 #3143 #3156 #3168 #3175 #3180 #3181 #3183 #3203 #3205 #3207 #3214 #3216 #3219 #3223 #3224 #3230 #3237 #3239 #3240 #3245 #3247 #3255 #3257 #3258 #3262 #3263 #3267 #3269 #3271 #3279 #3283 #3289 #3290 #3295)


Release 1.9 - 10/22/2020
------------------------

Major updates
^^^^^^^^^^^^^

Neural architecture search
""""""""""""""""""""""""""


* Support regularized evolution algorithm for NAS scenario (#2802)
* Add NASBench201 in search space zoo (#2766)

Model compression
"""""""""""""""""


* AMC pruner improvement: support resnet, support reproduction of the experiments (default parameters in our example code) in AMC paper (#2876 #2906)
* Support constraint-aware on some of our pruners to improve model compression efficiency (#2657)
* Support "tf.keras.Sequential" in model compression for TensorFlow (#2887)
* Support customized op in the model flops counter (#2795)
* Support quantizing bias in QAT quantizer (#2914)

Training service
""""""""""""""""


* Support configuring python environment using "preCommand" in remote mode (#2875)
* Support AML training service in Windows (#2882)
* Support reuse mode for remote training service (#2923)

WebUI & nnictl
""""""""""""""


* The "Overview" page on WebUI is redesigned with new layout (#2914)
* Upgraded node, yarn and FabricUI, and enabled Eslint (#2894 #2873 #2744)
* Add/Remove columns in hyper-parameter chart and trials table in "Trials detail" page (#2900)
* JSON format utility beautify on WebUI (#2863)
* Support nnictl command auto-completion (#2857)

UT & IT
^^^^^^^


* Add integration test for experiment import and export (#2878)
* Add integration test for user installed builtin tuner (#2859)
* Add unit test for nnictl (#2912)

Documentation
^^^^^^^^^^^^^


* Refactor of the document for model compression (#2919)

Bug fixes
^^^^^^^^^


* Bug fix of naïve evolution tuner, correctly deal with trial fails (#2695)
* Resolve the warning "WARNING (nni.protocol) IPC pipeline not exists, maybe you are importing tuner/assessor from trial code?" (#2864)
* Fix search space issue in experiment save/load (#2886)
* Fix bug in experiment import data (#2878)
* Fix annotation in remote mode (python 3.8 ast update issue) (#2881)
* Support boolean type for "choice" hyper-parameter when customizing trial configuration on WebUI (#3003)

Release 1.8 - 8/27/2020
-----------------------

Major updates
^^^^^^^^^^^^^

Training service
""""""""""""""""


* Access trial log directly on WebUI (local mode only) (#2718)
* Add OpenPAI trial job detail link (#2703)
* Support GPU scheduler in reusable environment (#2627) (#2769)
* Add timeout for ``web_channel`` in ``trial_runner`` (#2710)
* Show environment error message in AzureML mode (#2724)
* Add more log information when copying data in OpenPAI mode (#2702)

WebUI, nnictl and nnicli
""""""""""""""""""""""""


* Improve hyper-parameter parallel coordinates plot (#2691) (#2759)
* Add pagination for trial job list (#2738) (#2773)
* Enable panel close when clicking overlay region (#2734)
* Remove support for Multiphase on WebUI (#2760)
* Support save and restore experiments (#2750)
* Add intermediate results in export result (#2706)
* Add `command <https://github.com/microsoft/nni/blob/v1.8/docs/en_US/Tutorial/Nnictl.md#nnictl-trial>`__ to list trial results with highest/lowest metrics (#2747)
* Improve the user experience of `nnicli <https://github.com/microsoft/nni/blob/v1.8/docs/en_US/nnicli_ref.md>`__ with `examples <https://github.com/microsoft/nni/blob/v1.8/examples/notebooks/retrieve_nni_info_with_python.ipynb>`__ (#2713)

Neural architecture search
""""""""""""""""""""""""""


* `Search space zoo: ENAS and DARTS <https://github.com/microsoft/nni/blob/v1.8/docs/en_US/NAS/SearchSpaceZoo.md>`__ (#2589)
* API to query intermediate results in NAS benchmark (#2728)

Model compression
"""""""""""""""""


* Support the List/Tuple Construct/Unpack operation for TorchModuleGraph (#2609)
* Model speedup improvement: Add support of DenseNet and InceptionV3 (#2719)
* Support the multiple successive tuple unpack operations (#2768)
* `Doc of comparing the performance of supported pruners <https://github.com/microsoft/nni/blob/v1.8/docs/en_US/CommunitySharings/ModelCompressionComparison.md>`__ (#2742)
* New pruners: `Sensitivity pruner <https://github.com/microsoft/nni/blob/v1.8/docs/en_US/Compressor/Pruner.md#sensitivity-pruner>`__ (#2684) and `AMC pruner <https://github.com/microsoft/nni/blob/v1.8/docs/en_US/Compressor/Pruner.md>`__ (#2573) (#2786)
* TensorFlow v2 support in model compression (#2755)

Backward incompatible changes
"""""""""""""""""""""""""""""


* Update the default experiment folder from ``$HOME/nni/experiments`` to ``$HOME/nni-experiments``. If you want to view the experiments created by previous NNI releases, you can move the experiments folders from  ``$HOME/nni/experiments`` to ``$HOME/nni-experiments`` manually. (#2686) (#2753)
* Dropped support for Python 3.5 and scikit-learn 0.20 (#2778) (#2777) (2783) (#2787) (#2788) (#2790)

Others
""""""


* Upgrade TensorFlow version in Docker image (#2732) (#2735) (#2720)

Examples
^^^^^^^^


* Remove gpuNum in assessor examples (#2641)

Documentation
^^^^^^^^^^^^^


* Improve customized tuner documentation (#2628)
* Fix several typos and grammar mistakes in documentation (#2637 #2638, thanks @tomzx)
* Improve AzureML training service documentation (#2631)
* Improve CI of Chinese translation (#2654)
* Improve OpenPAI training service documentation (#2685)
* Improve documentation of community sharing (#2640)
* Add tutorial of Colab support (#2700)
* Improve documentation structure for model compression (#2676)

Bug fixes
^^^^^^^^^


* Fix mkdir error in training service (#2673)
* Fix bug when using chmod in remote training service (#2689)
* Fix dependency issue by making ``_graph_utils`` imported inline (#2675)
* Fix mask issue in ``SimulatedAnnealingPruner`` (#2736)
* Fix intermediate graph zooming issue (#2738)
* Fix issue when dict is unordered when querying NAS benchmark (#2728)
* Fix import issue for gradient selector dataloader iterator (#2690)
* Fix support of adding tens of machines in remote training service (#2725)
* Fix several styling issues in WebUI (#2762 #2737)
* Fix support of unusual types in metrics including NaN and Infinity (#2782)
* Fix nnictl experiment delete (#2791)

Release 1.7 - 7/8/2020
----------------------

Major Features
^^^^^^^^^^^^^^

Training Service
""""""""""""""""


* Support AML(Azure Machine Learning) platform as NNI training service.
* OpenPAI job can be reusable. When a trial is completed, the OpenPAI job won't stop, and wait next trial. `refer to reuse flag in OpenPAI config <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/PaiMode.md#openpai-configurations>`__.
* `Support ignoring files and folders in code directory with .nniignore when uploading code directory to training service <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/Overview.md#how-to-use-training-service>`__.

Neural Architecture Search (NAS)
""""""""""""""""""""""""""""""""


* 
  `Provide NAS Open Benchmarks (NasBench101, NasBench201, NDS) with friendly APIs <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/Benchmarks.md>`__.

* 
  `Support Classic NAS (i.e., non-weight-sharing mode) on TensorFlow 2.X <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/ClassicNas.md>`__.

Model Compression
"""""""""""""""""


* Improve Model Speedup: track more dependencies among layers and automatically resolve mask conflict, support the speedup of pruned resnet.
* Added new pruners, including three auto model pruning algorithms: `NetAdapt Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#netadapt-pruner>`__\ , `SimulatedAnnealing Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#simulatedannealing-pruner>`__\ , `AutoCompress Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#autocompress-pruner>`__\ , and `ADMM Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#admm-pruner>`__.
* Added `model sensitivity analysis tool <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/CompressionUtils.md>`__ to help users find the sensitivity of each layer to the pruning.
* 
  `Easy flops calculation for model compression and NAS <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/CompressionUtils.md#model-flops-parameters-counter>`__.

* 
  Update lottery ticket pruner to export winning ticket.

Examples
""""""""


* Automatically optimize tensor operators on NNI with a new `customized tuner OpEvo <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrialExample/OpEvoExamples.md>`__.

Built-in tuners/assessors/advisors
""""""""""""""""""""""""""""""""""


* `Allow customized tuners/assessor/advisors to be installed as built-in algorithms <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Tutorial/InstallCustomizedAlgos.md>`__.

WebUI
"""""


* Support visualizing nested search space more friendly.
* Show trial's dict keys in hyper-parameter graph.
* Enhancements to trial duration display.

Others
""""""


* Provide utility function to merge parameters received from NNI
* Support setting paiStorageConfigName in pai mode

Documentation
^^^^^^^^^^^^^


* Improve `documentation for model compression <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Overview.md>`__
* Improve `documentation <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/Benchmarks.md>`__
  and `examples <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/BenchmarksExample.ipynb>`__ for NAS benchmarks.
* Improve `documentation for AzureML training service <https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/AMLMode.md>`__
* Homepage migration to readthedoc.

Bug Fixes
^^^^^^^^^


* Fix bug for model graph with shared nn.Module
* Fix nodejs OOM when ``make build``
* Fix NASUI bugs
* Fix duration and intermediate results pictures update issue.
* Fix minor WebUI table style issues.

Release 1.6 - 5/26/2020
-----------------------

Major Features
^^^^^^^^^^^^^^

New Features and improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Improve IPC limitation to 100W
* improve code storage upload logic among trials in non-local platform
* support ``__version__`` for SDK version
* support windows dev intall

Web UI
^^^^^^


* Show trial error message
* finalize homepage layout
* Refactor overview's best trials module
* Remove multiphase from webui
* add tooltip for trial concurrency in the overview page
* Show top trials for hyper-parameter graph

HPO Updates
^^^^^^^^^^^


* Improve PBT on failure handling and support experiment resume for PBT

NAS Updates
^^^^^^^^^^^


* NAS support for TensorFlow 2.0 (preview) `TF2.0 NAS examples <https://github.com/microsoft/nni/tree/v1.6/examples/nas/naive-tf>`__
* Use OrderedDict for LayerChoice
* Prettify the format of export
* Replace layer choice with selected module after applied fixed architecture

Model Compression Updates
^^^^^^^^^^^^^^^^^^^^^^^^^


* Model compression PyTorch 1.4 support

Training Service Updates
^^^^^^^^^^^^^^^^^^^^^^^^


* update pai yaml merge logic
* support windows as remote machine in remote mode `Remote Mode <https://github.com/microsoft/nni/blob/v1.6/docs/en_US/TrainingService/RemoteMachineMode.md#windows>`__

Bug Fix
^^^^^^^


* fix dev install
* SPOS example crash when the checkpoints do not have state_dict
* Fix table sort issue when experiment had failed trial
* Support multi python env (conda, pyenv etc)

Release 1.5 - 4/13/2020
-----------------------

New Features and Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hyper-Parameter Optimizing
^^^^^^^^^^^^^^^^^^^^^^^^^^


* New tuner: `Population Based Training (PBT) <https://github.com/microsoft/nni/blob/v1.5/docs/en_US/Tuner/PBTTuner.md>`__
* Trials can now report infinity and NaN as result

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^


* New NAS algorithm: `TextNAS <https://github.com/microsoft/nni/blob/v1.5/docs/en_US/NAS/TextNAS.md>`__
* ENAS and DARTS now support `visualization <https://github.com/microsoft/nni/blob/v1.5/docs/en_US/NAS/Visualization.md>`__ through web UI.

Model Compression
^^^^^^^^^^^^^^^^^


* New Pruner: `GradientRankFilterPruner <https://github.com/microsoft/nni/blob/v1.5/docs/en_US/Compression/Pruner.md#gradientrankfilterpruner>`__
* Compressors will validate configuration by default
* Refactor: Adding optimizer as an input argument of pruner, for easy support of DataParallel and more efficient iterative pruning. This is a broken change for the usage of iterative pruning algorithms.
* Model compression examples are refactored and improved
* Added documentation for `implementing compressing algorithm <https://github.com/microsoft/nni/blob/v1.5/docs/en_US/Compression/Framework.md>`__

Training Service
^^^^^^^^^^^^^^^^


* Kubeflow now supports pytorchjob crd v1 (thanks external contributor @jiapinai)
* Experimental `DLTS <https://github.com/microsoft/nni/blob/v1.5/docs/en_US/TrainingService/DLTSMode.md>`__ support

Overall Documentation Improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Documentation is significantly improved on grammar, spelling, and wording (thanks external contributor @AHartNtkn)

Fixed Bugs
^^^^^^^^^^


* ENAS cannot have more than one LSTM layers (thanks external contributor @marsggbo)
* NNI manager's timers will never unsubscribe (thanks external contributor @guilhermehn)
* NNI manager may exhaust head memory (thanks external contributor @Sundrops)
* Batch tuner does not support customized trials (#2075)
* Experiment cannot be killed if it failed on start (#2080)
* Non-number type metrics break web UI (#2278)
* A bug in lottery ticket pruner
* Other minor glitches

Release 1.4 - 2/19/2020
-----------------------

Major Features
^^^^^^^^^^^^^^

Neural Architecture Search
^^^^^^^^^^^^^^^^^^^^^^^^^^


* Support `C-DARTS <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/CDARTS.md>`__ algorithm and add `the example <https://github.com/microsoft/nni/tree/v1.4/examples/nas/cdarts>`__ using it
* Support a preliminary version of `ProxylessNAS <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/Proxylessnas.md>`__ and the corresponding `example <https://github.com/microsoft/nni/tree/v1.4/examples/nas/proxylessnas>`__
* Add unit tests for the NAS framework

Model Compression
^^^^^^^^^^^^^^^^^


* Support DataParallel for compressing models, and provide `an example <https://github.com/microsoft/nni/blob/v1.4/examples/model_compress/multi_gpu.py>`__ of using DataParallel
* Support `model speedup <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Compressor/ModelSpeedup.md>`__ for compressed models, in Alpha version

Training Service
^^^^^^^^^^^^^^^^


* Support complete PAI configurations by allowing users to specify PAI config file path
* Add example config yaml files for the new PAI mode (i.e., paiK8S)
* Support deleting experiments using sshkey in remote mode (thanks external contributor @tyusr)

WebUI
^^^^^


* WebUI refactor: adopt fabric framework

Others
^^^^^^


* Support running `NNI experiment at foreground <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Tutorial/Nnictl.md#manage-an-experiment>`__\ , i.e., ``--foreground`` argument in ``nnictl create/resume/view``
* Support canceling the trials in UNKNOWN state
* Support large search space whose size could be up to 50mb (thanks external contributor @Sundrops)

Documentation
^^^^^^^^^^^^^


* Improve `the index structure <https://nni.readthedocs.io/en/latest/>`__ of NNI readthedocs
* Improve `documentation for NAS <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/NasGuide.md>`__
* Improve documentation for `the new PAI mode <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/TrainingService/PaiMode.md>`__
* Add QuickStart guidance for `NAS <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/QuickStart.md>`__ and `model compression <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Compressor/QuickStart.md>`__
* Improve documentation for `the supported EfficientNet <https://github.com/microsoft/nni/blob/v1.4/docs/en_US/TrialExample/EfficientNet.md>`__

Bug Fixes
^^^^^^^^^


* Correctly support NaN in metric data, JSON compliant
* Fix the out-of-range bug of ``randint`` type in search space
* Fix the bug of wrong tensor device when exporting onnx model in model compression
* Fix incorrect handling of nnimanagerIP in the new PAI mode (i.e., paiK8S)

Release 1.3 - 12/30/2019
------------------------

Major Features
^^^^^^^^^^^^^^

Neural Architecture Search Algorithms Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `Single Path One Shot <https://github.com/microsoft/nni/tree/v1.3/examples/nas/spos/>`__ algorithm and the example using it

Model Compression Algorithms Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `Knowledge Distillation <https://github.com/microsoft/nni/blob/v1.3/docs/en_US/TrialExample/KDExample.md>`__ algorithm and the example using itExample
* Pruners

  * `L2Filter Pruner <https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#3-l2filter-pruner>`__
  * `ActivationAPoZRankFilterPruner <https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#1-activationapozrankfilterpruner>`__
  * `ActivationMeanRankFilterPruner <https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#2-activationmeanrankfilterpruner>`__

* `BNN Quantizer <https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Quantizer.md#bnn-quantizer>`__

Training Service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* 
  NFS Support for PAI

    Instead of using HDFS as default storage, since OpenPAI v0.11, OpenPAI can have NFS or AzureBlob or other storage as default storage. In this release, NNI extended the support for this recent change made by OpenPAI, and could integrate with OpenPAI v0.11 or later version with various default storage.

* 
  Kubeflow update adoption

    Adopted the Kubeflow 0.7's new supports for tf-operator.

Engineering (code and build automation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Enforced `ESLint <https://eslint.org/>`__ on static code analysis.

Small changes & Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^^^


* correctly recognize builtin tuner and customized tuner
* logging in dispatcher base
* fix the bug where tuner/assessor's failure sometimes kills the experiment.
* Fix local system as remote machine `issue <https://github.com/microsoft/nni/issues/1852>`__
* de-duplicate trial configuration in smac tuner `ticket <https://github.com/microsoft/nni/issues/1364>`__

Release 1.2 - 12/02/2019
------------------------

Major Features
^^^^^^^^^^^^^^


* `Feature Engineering <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/Overview.md>`__

  * New feature engineering interface
  * Feature selection algorithms: `Gradient feature selector <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GradientFeatureSelector.md>`__ & `GBDT selector <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GBDTSelector.md>`__
  * `Examples for feature engineering <https://github.com/microsoft/nni/tree/v1.2/examples/feature_engineering>`__

* Neural Architecture Search (NAS) on NNI

  * `New NAS interface <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/NasInterface.md>`__
  * NAS algorithms: `ENAS <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#enas>`__\ , `DARTS <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#darts>`__\ , `P-DARTS <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#p-darts>`__ (in PyTorch)
  * NAS in classic mode (each trial runs independently)

* Model compression

  * `New model pruning algorithms <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md>`__\ : lottery ticket pruning approach, L1Filter pruner, Slim pruner, FPGM pruner
  * `New model quantization algorithms <https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md>`__\ : QAT quantizer, DoReFa quantizer
  * Support the API for exporting compressed model.

* Training Service

  * Support OpenPAI token authentication

* Examples:

  * `An example to automatically tune rocksdb configuration with NNI <https://github.com/microsoft/nni/tree/v1.2/examples/trials/systems/rocksdb-fillrandom>`__.
  * `A new MNIST trial example supports tensorflow 2.0 <https://github.com/microsoft/nni/tree/v1.2/examples/trials/mnist-tfv2>`__.

* Engineering Improvements

  * For remote training service,  trial jobs require no GPU are now scheduled with round-robin policy instead of random.
  * Pylint rules added to check pull requests, new pull requests need to comply with these `pylint rules <https://github.com/microsoft/nni/blob/v1.2/pylintrc>`__.

* Web Portal & User Experience

  * Support user to add customized trial.
  * User can zoom out/in in detail graphs, except Hyper-parameter.

* Documentation

  * Improved NNI API documentation with more API docstring.

Bug fix
^^^^^^^


* Fix the table sort issue when failed trials haven't metrics. -Issue #1773
* Maintain selected status(Maximal/Minimal) when the page switched. -PR#1710
* Make hyper-parameters graph's default metric yAxis more accurate. -PR#1736
* Fix GPU script permission issue. -Issue #1665

Release 1.1 - 10/23/2019
------------------------

Major Features
^^^^^^^^^^^^^^


* New tuner: `PPO Tuner <https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tuner/PPOTuner.md>`__
* `View stopped experiments <https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/Nnictl.md#view>`__
* Tuners can now use dedicated GPU resource (see ``gpuIndices`` in `tutorial <https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/ExperimentConfig.md>`__ for details)
* Web UI improvements

  * Trials detail page can now list hyperparameters of each trial, as well as their start and end time (via "add column")
  * Viewing huge experiment is now less laggy

* More examples

  * `EfficientNet PyTorch example <https://github.com/ultmaster/EfficientNet-PyTorch>`__
  * `Cifar10 NAS example <https://github.com/microsoft/nni/blob/v1.1/examples/trials/nas_cifar10/README.md>`__

* `Model compression toolkit - Alpha release <https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Compressor/Overview.md>`__\ : We are glad to announce the alpha release for model compression toolkit on top of NNI, it's still in the experiment phase which might evolve based on usage feedback. We'd like to invite you to use, feedback and even contribute

Fixed Bugs
^^^^^^^^^^


* Multiphase job hangs when search space exhuasted (issue #1204)
* ``nnictl`` fails when log not available (issue #1548)

Release 1.0 - 9/2/2019
----------------------

Major Features
^^^^^^^^^^^^^^


* 
  Tuners and Assessors


  * Support Auto-Feature generator & selection    -Issue#877  -PR #1387

    * Provide auto feature interface
    * Tuner based on beam search
    * `Add Pakdd example <https://github.com/microsoft/nni/tree/v1.0/examples/trials/auto-feature-engineering>`__

  * Add a parallel algorithm to improve the performance of TPE with large concurrency.  -PR #1052
  * Support multiphase for hyperband    -PR #1257

* 
  Training Service


  * Support private docker registry   -PR #755


  * Engineering Improvements

    * Python wrapper for rest api, support retrieve the values of the metrics in a programmatic way  PR #1318
    * New python API : get_experiment_id(), get_trial_id()  -PR #1353   -Issue #1331 & -Issue#1368
    * Optimized NAS Searchspace  -PR #1393

      * Unify NAS search space with _type -- "mutable_type"e
      * Update random search tuner

    * Set gpuNum as optional      -Issue #1365
    * Remove outputDir and dataDir configuration in PAI mode   -Issue #1342
    * When creating a trial in Kubeflow mode, codeDir will no longer be copied to logDir   -Issue #1224

* 
  Web Portal & User Experience


  * Show the best metric curve during search progress in WebUI  -Issue #1218
  * Show the current number of parameters list in multiphase experiment   -Issue1210  -PR #1348
  * Add "Intermediate count" option in AddColumn.      -Issue #1210
  * Support search parameters value in WebUI     -Issue #1208
  * Enable automatic scaling of axes for metric value  in default metric graph   -Issue #1360
  * Add a detailed documentation link to the nnictl command in the command prompt    -Issue #1260
  * UX improvement for showing Error log   -Issue #1173

* 
  Documentation


  * Update the docs structure  -Issue #1231
  * (deprecated) Multi phase document improvement   -Issue #1233  -PR #1242

    * Add configuration example

  * `WebUI description improvement <Tutorial/WebUI.rst>`__  -PR #1419

Bug fix
^^^^^^^


* (Bug fix)Fix the broken links in 0.9 release  -Issue #1236
* (Bug fix)Script for auto-complete
* (Bug fix)Fix pipeline issue that it only check exit code of last command in a script.  -PR #1417
* (Bug fix)quniform fors tuners    -Issue #1377
* (Bug fix)'quniform' has different meaning beween GridSearch and other tuner.   -Issue #1335
* (Bug fix)"nnictl experiment list" give the status of a "RUNNING" experiment as "INITIALIZED" -PR #1388
* (Bug fix)SMAC cannot be installed if nni is installed in dev mode    -Issue #1376
* (Bug fix)The filter button of the intermediate result cannot be clicked   -Issue #1263
* (Bug fix)API "/api/v1/nni/trial-jobs/xxx" doesn't show a trial's all parameters in multiphase experiment    -Issue #1258
* (Bug fix)Succeeded trial doesn't have final result but webui show ×××(FINAL)  -Issue #1207
* (Bug fix)IT for nnictl stop -Issue #1298
* (Bug fix)fix security warning
* (Bug fix)Hyper-parameter page broken  -Issue #1332
* (Bug fix)Run flake8 tests to find Python syntax errors and undefined names -PR #1217

Release 0.9 - 7/1/2019
----------------------

Major Features
^^^^^^^^^^^^^^


* General NAS programming interface

  * Add ``enas-mode``  and ``oneshot-mode`` for NAS interface: `PR #1201 <https://github.com/microsoft/nni/pull/1201#issue-291094510>`__

* 
  `Gaussian Process Tuner with Matern kernel <Tuner/GPTuner.rst>`__

* 
  (deprecated) Multiphase experiment supports


  * Added new training service support for multiphase experiment: PAI mode supports multiphase experiment since v0.9.
  * Added multiphase capability for the following builtin tuners:

    * TPE, Random Search, Anneal, Naïve Evolution, SMAC, Network Morphism, Metis Tuner.

* 
  Web Portal


  * Enable trial comparation in Web Portal. For details, refer to `View trials status <Tutorial/WebUI.rst>`__
  * Allow users to adjust rendering interval of Web Portal. For details, refer to `View Summary Page <Tutorial/WebUI.rst>`__
  * show intermediate results more friendly. For details, refer to `View trials status <Tutorial/WebUI.rst>`__

* `Commandline Interface <Tutorial/Nnictl.rst>`__

  * ``nnictl experiment delete``\ : delete one or all experiments, it includes log, result, environment information and cache. It uses to delete useless experiment result, or save disk space.
  * ``nnictl platform clean``\ : It uses to clean up disk on a target platform. The provided YAML file includes the information of target platform, and it follows the same schema as the NNI configuration file.

Bug fix and other changes
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Tuner Installation Improvements: add `sklearn <https://scikit-learn.org/stable/>`__ to nni dependencies.
* (Bug Fix) Failed to connect to PAI http code - `Issue #1076 <https://github.com/microsoft/nni/issues/1076>`__
* (Bug Fix) Validate file name for PAI platform - `Issue #1164 <https://github.com/microsoft/nni/issues/1164>`__
* (Bug Fix) Update GMM evaluation in Metis Tuner
* (Bug Fix) Negative time number rendering in Web Portal - `Issue #1182 <https://github.com/microsoft/nni/issues/1182>`__\ , `Issue #1185 <https://github.com/microsoft/nni/issues/1185>`__
* (Bug Fix) Hyper-parameter not shown correctly in WebUI when there is only one hyper parameter - `Issue #1192 <https://github.com/microsoft/nni/issues/1192>`__

Release 0.8 - 6/4/2019
----------------------

Major Features
^^^^^^^^^^^^^^


* Support NNI on Windows for OpenPAI/Remote mode

  * NNI running on windows for remote mode
  * NNI running on windows for OpenPAI mode

* Advanced features for using GPU

  * Run multiple trial jobs on the same GPU for local and remote mode
  * Run trial jobs on the GPU running non-NNI jobs

* Kubeflow v1beta2 operator

  * Support Kubeflow TFJob/PyTorchJob v1beta2

* `General NAS programming interface <https://github.com/microsoft/nni/blob/v0.8/docs/en_US/GeneralNasInterfaces.md>`__

  * Provide NAS programming interface for users to easily express their neural architecture search space through NNI annotation
  * Provide a new command ``nnictl trial codegen`` for debugging the NAS code
  * Tutorial of NAS programming interface, example of NAS on MNIST, customized random tuner for NAS

* Support resume tuner/advisor's state for experiment resume
* For experiment resume, tuner/advisor will be resumed by replaying finished trial data
* Web Portal

  * Improve the design of copying trial's parameters
  * Support 'randint' type in hyper-parameter graph
  * Use should ComponentUpdate to avoid unnecessary render

Bug fix and other changes
^^^^^^^^^^^^^^^^^^^^^^^^^


* Bug fix that ``nnictl update`` has inconsistent command styles
* Support import data for SMAC tuner
* Bug fix that experiment state transition from ERROR back to RUNNING
* Fix bug of table entries
* Nested search space refinement
* Refine 'randint' type and support lower bound
* `Comparison of different hyper-parameter tuning algorithm <CommunitySharings/HpoComparison.rst>`__
* `Comparison of NAS algorithm <CommunitySharings/NasComparison.rst>`__
* `NNI practice on Recommenders <CommunitySharings/RecommendersSvd.rst>`__

Release 0.7 - 4/29/2018
-----------------------

Major Features
^^^^^^^^^^^^^^


* `Support NNI on Windows <Tutorial/InstallationWin.rst>`__

  * NNI running on windows for local mode

* `New advisor: BOHB <Tuner/BohbAdvisor.rst>`__

  * Support a new advisor BOHB, which is a robust and efficient hyperparameter tuning algorithm, combines the advantages of Bayesian optimization and Hyperband

* `Support import and export experiment data through nnictl <Tutorial/Nnictl.rst>`__

  * Generate analysis results report after the experiment execution
  * Support import data to tuner and advisor for tuning

* `Designated gpu devices for NNI trial jobs <Tutorial/ExperimentConfig.rst#localConfig>`__

  * Specify GPU devices for NNI trial jobs by gpuIndices configuration, if gpuIndices is set in experiment configuration file, only the specified GPU devices are used for NNI trial jobs.

* Web Portal enhancement

  * Decimal format of metrics other than default on the Web UI
  * Hints in WebUI about Multi-phase
  * Enable copy/paste for hyperparameters as python dict
  * Enable early stopped trials data for tuners.

* NNICTL provide better error message

  * nnictl provide more meaningful error message for YAML file format error

Bug fix
^^^^^^^


* Unable to kill all python threads after nnictl stop in async dispatcher mode
* nnictl --version does not work with make dev-install
* All trail jobs status stays on 'waiting' for long time on OpenPAI platform

Release 0.6 - 4/2/2019
----------------------

Major Features
^^^^^^^^^^^^^^


* `Version checking <TrainingService/PaiMode.rst>`__

  * check whether the version is consistent between nniManager and trialKeeper

* `Report final metrics for early stop job <https://github.com/microsoft/nni/issues/776>`__

  * If includeIntermediateResults is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result. The default value of includeIntermediateResults is false.

* `Separate Tuner/Assessor <https://github.com/microsoft/nni/issues/841>`__

  * Adds two pipes to separate message receiving channels for tuner and assessor.

* Make log collection feature configurable
* Add intermediate result graph for all trials

Bug fix
^^^^^^^


* `Add shmMB config key for OpenPAI <https://github.com/microsoft/nni/issues/842>`__
* Fix the bug that doesn't show any result if metrics is dict
* Fix the number calculation issue for float types in hyperband
* Fix a bug in the search space conversion in SMAC tuner
* Fix the WebUI issue when parsing experiment.json with illegal format
* Fix cold start issue in Metis Tuner

Release 0.5.2 - 3/4/2019
------------------------

Improvements
^^^^^^^^^^^^


* Curve fitting assessor performance improvement.

Documentation
^^^^^^^^^^^^^


* Chinese version document: https://nni.readthedocs.io/zh/latest/
* Debuggability/serviceability document: https://nni.readthedocs.io/en/latest/Tutorial/HowToDebug.html
* Tuner assessor reference: https://nni.readthedocs.io/en/latest/sdk_reference.html

Bug Fixes and Other Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Fix a race condition bug that does not store trial job cancel status correctly.
* Fix search space parsing error when using SMAC tuner.
* Fix cifar10 example broken pipe issue.
* Add unit test cases for nnimanager and local training service.
* Add integration test azure pipelines for remote machine, OpenPAI and kubeflow training services.
* Support Pylon in OpenPAI webhdfs client.

Release 0.5.1 - 1/31/2018
-------------------------

Improvements
^^^^^^^^^^^^


* Making `log directory <https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md>`__ configurable
* Support `different levels of logs <https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md>`__\ , making it easier for debugging

Documentation
^^^^^^^^^^^^^


* Reorganized documentation & New Homepage Released: https://nni.readthedocs.io/en/latest/

Bug Fixes and Other Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Fix the bug of installation in python virtualenv, and refactor the installation logic
* Fix the bug of HDFS access failure on OpenPAI mode after OpenPAI is upgraded.
* Fix the bug that sometimes in-place flushed stdout makes experiment crash

Release 0.5.0 - 01/14/2019
--------------------------

Major Features
^^^^^^^^^^^^^^

New tuner and assessor supports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Support `Metis tuner <Tuner/MetisTuner.rst>`__ as a new NNI tuner. Metis algorithm has been proofed to be well performed for **online** hyper-parameter tuning.
* Support `ENAS customized tuner <https://github.com/countif/enas_nni>`__\ , a tuner contributed by github community user, is an algorithm for neural network search, it could learn neural network architecture via reinforcement learning and serve a better performance than NAS.
* Support `Curve fitting assessor <Assessor/CurvefittingAssessor.rst>`__ for early stop policy using learning curve extrapolation.
* Advanced Support of `Weight Sharing <https://github.com/microsoft/nni/blob/v0.5/docs/AdvancedNAS.md>`__\ : Enable weight sharing for NAS tuners, currently through NFS.

Training Service Enhancement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `FrameworkController Training service <TrainingService/FrameworkControllerMode.rst>`__\ : Support run experiments using frameworkcontroller on kubernetes

  * FrameworkController is a Controller on kubernetes that is general enough to run (distributed) jobs with various machine learning frameworks, such as tensorflow, pytorch, MXNet.
  * NNI provides unified and simple specification for job definition.
  * MNIST example for how to use FrameworkController.

User Experience improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* A better trial logging support for NNI experiments in OpenPAI, Kubeflow and FrameworkController mode:

  * An improved logging architecture to send stdout/stderr of trials to NNI manager via Http post. NNI manager will store trial's stdout/stderr messages in local log file.
  * Show the link for trial log file on WebUI.

* Support to show final result's all key-value pairs.

Release 0.4.1 - 12/14/2018
--------------------------

Major Features
^^^^^^^^^^^^^^

New tuner supports
^^^^^^^^^^^^^^^^^^


* Support `network morphism <Tuner/NetworkmorphismTuner.rst>`__ as a new tuner

Training Service improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* Migrate `Kubeflow training service <TrainingService/KubeflowMode.rst>`__\ 's dependency from kubectl CLI to `Kubernetes API <https://kubernetes.io/docs/concepts/overview/kubernetes-api/>`__ client
* `Pytorch-operator <https://github.com/kubeflow/pytorch-operator>`__ support for Kubeflow training service
* Improvement on local code files uploading to OpenPAI HDFS
* Fixed OpenPAI integration WebUI bug: WebUI doesn't show latest trial job status, which is caused by OpenPAI token expiration

NNICTL improvements
^^^^^^^^^^^^^^^^^^^


* Show version information both in nnictl and WebUI. You can run **nnictl -v** to show your current installed NNI version

WebUI improvements
^^^^^^^^^^^^^^^^^^


* Enable modify concurrency number during experiment
* Add feedback link to NNI github 'create issue' page
* Enable customize top 10 trials regarding to metric numbers (largest or smallest)
* Enable download logs for dispatcher & nnimanager
* Enable automatic scaling of axes for metric number
* Update annotation to support displaying real choice in searchspace

New examples
^^^^^^^^^^^^


* `FashionMnist <https://github.com/microsoft/nni/tree/v0.5/examples/trials/network_morphism>`__\ , work together with network morphism tuner
* `Distributed MNIST example <https://github.com/microsoft/nni/tree/v0.5/examples/trials/mnist-distributed-pytorch>`__ written in PyTorch

Release 0.4 - 12/6/2018
-----------------------

Major Features
^^^^^^^^^^^^^^


* `Kubeflow Training service <TrainingService/KubeflowMode.rst>`__

  * Support tf-operator
  * `Distributed trial example <https://github.com/microsoft/nni/tree/v0.4/examples/trials/mnist-distributed/dist_mnist.py>`__ on Kubeflow

* `Grid search tuner <Tuner/GridsearchTuner.rst>`__
* `Hyperband tuner <Tuner/HyperbandAdvisor.rst>`__
* Support launch NNI experiment on MAC
* WebUI

  * UI support for hyperband tuner
  * Remove tensorboard button
  * Show experiment error message
  * Show line numbers in search space and trial profile
  * Support search a specific trial by trial number
  * Show trial's hdfsLogPath
  * Download experiment parameters

Others
^^^^^^


* Asynchronous dispatcher
* Docker file update, add pytorch library
* Refactor 'nnictl stop' process, send SIGTERM to nni manager process, rather than calling stop Rest API.
* OpenPAI training service bug fix

  * Support NNI Manager IP configuration(nniManagerIp) in OpenPAI cluster config file, to fix the issue that user’s machine has no eth0 device
  * File number in codeDir is capped to 1000 now, to avoid user mistakenly fill root dir for codeDir
  * Don’t print useless ‘metrics is empty’ log in OpenPAI job’s stdout. Only print useful message once new metrics are recorded, to reduce confusion when user checks OpenPAI trial’s output for debugging purpose
  * Add timestamp at the beginning of each log entry in trial keeper.

Release 0.3.0 - 11/2/2018
-------------------------

NNICTL new features and updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  Support running multiple experiments simultaneously.

  Before v0.3, NNI only supports running single experiment once a time. After this release, users are able to run multiple experiments simultaneously. Each experiment will require a unique port, the 1st experiment will be set to the default port as previous versions. You can specify a unique port for the rest experiments as below:

  .. code-block:: text

     nnictl create --port 8081 --config <config file path>

* 
  Support updating max trial number.
  use ``nnictl update --help`` to learn more. Or refer to `NNICTL Spec <Tutorial/Nnictl.rst>`__ for the fully usage of NNICTL.

API new features and updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  :raw-html:`<span style="color:red">**breaking change**</span>`\ : nn.get_parameters() is refactored to nni.get_next_parameter. All examples of prior releases can not run on v0.3, please clone nni repo to get new examples. If you had applied NNI to your own codes, please update the API accordingly.

* 
  New API **nni.get_sequence_id()**.
  Each trial job is allocated a unique sequence number, which can be retrieved by nni.get_sequence_id() API.

  .. code-block:: bash

     git clone -b v0.3 https://github.com/microsoft/nni.git

* 
  **nni.report_final_result(result)** API supports more data types for result parameter.

  It can be of following types:


  * int
  * float
  * A python dict containing 'default' key, the value of 'default' key should be of type int or float. The dict can contain any other key value pairs.

New tuner support
^^^^^^^^^^^^^^^^^


* **Batch Tuner** which iterates all parameter combination, can be used to submit batch trial jobs.

New examples
^^^^^^^^^^^^


* 
  A NNI Docker image for public usage:

  .. code-block:: bash

     docker pull msranni/nni:latest

* 
  New trial example: `NNI Sklearn Example <https://github.com/microsoft/nni/tree/v0.3/examples/trials/sklearn>`__

* New competition example: `Kaggle Competition TGS Salt Example <https://github.com/microsoft/nni/tree/v0.3/examples/trials/kaggle-tgs-salt>`__

Others
^^^^^^


* UI refactoring, refer to `WebUI doc <Tutorial/WebUI.rst>`__ for how to work with the new UI.
* Continuous Integration: NNI had switched to Azure pipelines

Release 0.2.0 - 9/29/2018
-------------------------

Major Features
^^^^^^^^^^^^^^


* Support `OpenPAI <https://github.com/microsoft/pai>`__ Training Platform (See `here <TrainingService/PaiMode.rst>`__ for instructions about how to submit NNI job in pai mode)

  * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
  * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking

* Support `SMAC <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ tuner (See `here <Tuner/SmacTuner.rst>`__ for instructions about how to use SMAC tuner)

  * `SMAC <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO to handle categorical parameters. The SMAC supported by NNI is a wrapper on `SMAC3 <https://github.com/automl/SMAC3>`__

* Support NNI installation on `conda <https://conda.io/docs/index.html>`__ and python virtual environment
* Others

  * Update ga squad example and related documentation
  * WebUI UX small enhancement and bug fix

Release 0.1.0 - 9/10/2018 (initial release)
-------------------------------------------

Initial release of Neural Network Intelligence (NNI).

Major Features
^^^^^^^^^^^^^^


* Installation and Deployment

  * Support pip install and source codes install
  * Support training services on local mode(including Multi-GPU mode) as well as multi-machines mode

* Tuners, Assessors and Trial

  * Support AutoML algorithms including:  hyperopt_tpe, hyperopt_annealing, hyperopt_random, and evolution_tuner
  * Support assessor(early stop) algorithms including: medianstop algorithm
  * Provide Python API for user defined tuners and assessors
  * Provide Python API for user to wrap trial code as NNI deployable codes

* Experiments

  * Provide a command line toolkit 'nnictl' for experiments management
  * Provide a WebUI for viewing experiments details and managing experiments

* Continuous Integration

  * Support CI by providing out-of-box integration with `travis-ci <https://github.com/travis-ci>`__ on ubuntu

* Others

  * Support simple GPU job scheduling
