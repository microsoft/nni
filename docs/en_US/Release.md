# ChangeLog

# Release 1.7 - 7/8/2020

## Major Features

### Training Service

* Support AML(Azure Machine Learning) platform as NNI training service.
* OpenPAI job can be reusable. When a trial is completed, the OpenPAI job won't stop, and wait next trial. [refer to reuse flag in OpenPAI config](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/PaiMode.md#openpai-configurations).
* [Support ignoring files and folders in code directory with .nniignore when uploading code directory to training service](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/Overview.md#how-to-use-training-service).

### Neural Architecture Search (NAS)

* [Provide NAS Open Benchmarks (NasBench101, NasBench201, NDS) with friendly APIs](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/Benchmarks.md).

* [Support Classic NAS (i.e., non-weight-sharing mode) on TensorFlow 2.X](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/ClassicNas.md).

### Model Compression

* Improve Model Speedup: track more dependencies among layers and automatically resolve mask conflict, support the speedup of pruned resnet.
* Added new pruners, including three auto model pruning algorithms: [NetAdapt Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#netadapt-pruner), [SimulatedAnnealing Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#simulatedannealing-pruner), [AutoCompress Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#autocompress-pruner), and [ADMM Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#admm-pruner).
* Added [model sensitivity analysis tool](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/CompressionUtils.md) to help users find the sensitivity of each layer to the pruning.
* [Easy flops calculation for model compression and NAS](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/CompressionUtils.md#model-flops-parameters-counter).

* Update lottery ticket pruner to export winning ticket.

### Examples

* Automatically optimize tensor operators on NNI with a new [customized tuner OpEvo](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrialExample/OpEvoExamples.md).

### Built-in tuners/assessors/advisors

* [Allow customized tuners/assessor/advisors to be installed as built-in algorithms](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Tutorial/InstallCustomizedAlgos.md).

### WebUI

* Support visualizing nested search space more friendly.
* Show trial's dict keys in hyper-parameter graph.
* Enhancements to trial duration display.

### Others

* Provide utility function to merge parameters received from NNI
* Support setting paiStorageConfigName in pai mode

## Documentation

* Improve [documentation for model compression](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Overview.md)
* Improve [documentation](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/Benchmarks.md)
and [examples](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/BenchmarksExample.ipynb) for NAS benchmarks.
* Improve [documentation for AzureML training service](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/AMLMode.md)
* Homepage migration to readthedoc.

## Bug Fixes

* Fix bug for model graph with shared nn.Module
* Fix nodejs OOM when `make build`
* Fix NASUI bugs
* Fix duration and intermediate results pictures update issue.
* Fix minor WebUI table style issues.

## Release 1.6 - 5/26/2020

### Major Features

#### New Features and improvement
* Improve IPC limitation to 100W
* improve code storage upload logic among trials in non-local platform
* support `__version__` for SDK version
* support windows dev intall

#### Web UI
* Show trial error message
* finalize homepage layout
* Refactor overview's best trials module
* Remove multiphase from webui
* add tooltip for trial concurrency in the overview page
* Show top trials for hyper-parameter graph

#### HPO Updates
* Improve PBT on failure handling and support experiment resume for PBT

#### NAS Updates
* NAS support for TensorFlow 2.0 (preview) [TF2.0 NAS examples](https://github.com/microsoft/nni/tree/master/examples/nas/naive-tf)
* Use OrderedDict for LayerChoice
* Prettify the format of export
* Replace layer choice with selected module after applied fixed architecture

#### Model Compression Updates
* Model compression PyTorch 1.4 support

#### Training Service Updates
* update pai yaml merge logic
* support windows as remote machine in remote mode [Remote Mode](https://github.com/microsoft/nni/blob/master/docs/en_US/TrainingService/RemoteMachineMode.md#windows)

### Bug Fix
* fix dev install
* SPOS example crash when the checkpoints do not have state_dict
* Fix table sort issue when experiment had failed trial
* Support multi python env (conda, pyenv etc)


## Release 1.5 - 4/13/2020

### New Features and Documentation

#### Hyper-Parameter Optimizing

* New tuner: [Population Based Training (PBT)](https://github.com/microsoft/nni/blob/master/docs/en_US/Tuner/PBTTuner.md)
* Trials can now report infinity and NaN as result

#### Neural Architecture Search

* New NAS algorithm: [TextNAS](https://github.com/microsoft/nni/blob/master/docs/en_US/NAS/TextNAS.md)
* ENAS and DARTS now support [visualization](https://github.com/microsoft/nni/blob/master/docs/en_US/NAS/Visualization.md) through web UI.

#### Model Compression

* New Pruner: [GradientRankFilterPruner](https://github.com/microsoft/nni/blob/master/docs/en_US/Compressor/Pruner.md#gradientrankfilterpruner)
* Compressors will validate configuration by default
* Refactor: Adding optimizer as an input argument of pruner, for easy support of DataParallel and more efficient iterative pruning. This is a broken change for the usage of iterative pruning algorithms.
* Model compression examples are refactored and improved
* Added documentation for [implementing compressing algorithm](https://github.com/microsoft/nni/blob/master/docs/en_US/Compressor/Framework.md)

#### Training Service

* Kubeflow now supports pytorchjob crd v1 (thanks external contributor @jiapinai)
* Experimental [DLTS](https://github.com/microsoft/nni/blob/master/docs/en_US/TrainingService/DLTSMode.md) support

#### Overall Documentation Improvement

* Documentation is significantly improved on grammar, spelling, and wording (thanks external contributor @AHartNtkn)

### Fixed Bugs

* ENAS cannot have more than one LSTM layers (thanks external contributor @marsggbo)
* NNI manager's timers will never unsubscribe (thanks external contributor @guilhermehn)
* NNI manager may exhaust head memory (thanks external contributor @Sundrops)
* Batch tuner does not support customized trials (#2075)
* Experiment cannot be killed if it failed on start (#2080)
* Non-number type metrics break web UI (#2278)
* A bug in lottery ticket pruner
* Other minor glitches

## Release 1.4 - 2/19/2020

### Major Features

#### Neural Architecture Search
* Support [C-DARTS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/CDARTS.md) algorithm and add [the example](https://github.com/microsoft/nni/tree/v1.4/examples/nas/cdarts) using it
* Support a preliminary version of [ProxylessNAS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/Proxylessnas.md) and the corresponding [example](https://github.com/microsoft/nni/tree/v1.4/examples/nas/proxylessnas)
* Add unit tests for the NAS framework

#### Model Compression
* Support DataParallel for compressing models, and provide [an example](https://github.com/microsoft/nni/blob/v1.4/examples/model_compress/multi_gpu.py) of using DataParallel
* Support [model speedup](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Compressor/ModelSpeedup.md) for compressed models, in Alpha version

#### Training Service
* Support complete PAI configurations by allowing users to specify PAI config file path
* Add example config yaml files for the new PAI mode (i.e., paiK8S)
* Support deleting experiments using sshkey in remote mode (thanks external contributor @tyusr)

#### WebUI
* WebUI refactor: adopt fabric framework

#### Others
* Support running [NNI experiment at foreground](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Tutorial/Nnictl#manage-an-experiment), i.e., `--foreground` argument in `nnictl create/resume/view`
* Support canceling the trials in UNKNOWN state
* Support large search space whose size could be up to 50mb (thanks external contributor @Sundrops)

### Documentation
* Improve [the index structure](https://nni.readthedocs.io/en/latest/) of NNI readthedocs
* Improve [documentation for NAS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/NasGuide.md)
* Improve documentation for [the new PAI mode](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/TrainingService/PaiMode.md)
* Add QuickStart guidance for [NAS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/QuickStart.md) and [model compression](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Compressor/QuickStart.md)
* Improve documentation for [the supported EfficientNet](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/TrialExample/EfficientNet.md)

### Bug Fixes
* Correctly support NaN in metric data, JSON compliant
* Fix the out-of-range bug of `randint` type in search space
* Fix the bug of wrong tensor device when exporting onnx model in model compression
* Fix incorrect handling of nnimanagerIP in the new PAI mode (i.e., paiK8S)


## Release 1.3 - 12/30/2019

### Major Features

#### Neural Architecture Search Algorithms Support
* [Single Path One Shot](https://github.com/microsoft/nni/tree/v1.3/examples/nas/spos/) algorithm and the example using it

#### Model Compression Algorithms Support
* [Knowledge Distillation](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/TrialExample/KDExample.md) algorithm and the example using itExample
* Pruners
    * [L2Filter Pruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#3-l2filter-pruner)
    * [ActivationAPoZRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#1-activationapozrankfilterpruner)
    * [ActivationMeanRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#2-activationmeanrankfilterpruner)
* [BNN Quantizer](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Quantizer.md#bnn-quantizer)
#### Training Service
* NFS Support for PAI

    Instead of using HDFS as default storage, since OpenPAI v0.11, OpenPAI can have NFS or AzureBlob or other storage as default storage. In this release, NNI extended the support for this recent change made by OpenPAI, and could integrate with OpenPAI v0.11 or later version with various default storage.

* Kubeflow update adoption

    Adopted the Kubeflow 0.7's new supports for tf-operator.

### Engineering (code and build automation)
* Enforced [ESLint](https://eslint.org/) on static code analysis.

### Small changes & Bug Fixes
* correctly recognize builtin tuner and customized tuner
* logging in dispatcher base
* fix the bug where tuner/assessor's failure sometimes kills the experiment.
* Fix local system as remote machine [issue](https://github.com/microsoft/nni/issues/1852)
* de-duplicate trial configuration in smac tuner [ticket](https://github.com/microsoft/nni/issues/1364)

## Release 1.2 - 12/02/2019

### Major Features
* [Feature Engineering](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/Overview.md)
  - New feature engineering interface
  - Feature selection algorithms: [Gradient feature selector](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GradientFeatureSelector.md) & [GBDT selector](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GBDTSelector.md)
  - [Examples for feature engineering](https://github.com/microsoft/nni/tree/v1.2/examples/feature_engineering)
* Neural Architecture Search (NAS) on NNI
  - [New NAS interface](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/NasInterface.md)
  - NAS algorithms: [ENAS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#enas), [DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#darts), [P-DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#p-darts) (in PyTorch)
  - NAS in classic mode (each trial runs independently)
* Model compression
  - [New model pruning algorithms](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md): lottery ticket pruning approach, L1Filter pruner, Slim pruner, FPGM pruner
  - [New model quantization algorithms](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md): QAT quantizer, DoReFa quantizer
  - Support the API for exporting compressed model.
* Training Service
  - Support OpenPAI token authentication
* Examples:
  - [An example to automatically tune rocksdb configuration with NNI](https://github.com/microsoft/nni/tree/v1.2/examples/trials/systems/rocksdb-fillrandom).
  - [A new MNIST trial example supports tensorflow 2.0](https://github.com/microsoft/nni/tree/v1.2/examples/trials/mnist-tfv2).
* Engineering Improvements
  - For remote training service,  trial jobs require no GPU are now scheduled with round-robin policy instead of random.
  - Pylint rules added to check pull requests, new pull requests need to comply with these [pylint rules](https://github.com/microsoft/nni/blob/v1.2/pylintrc).
* Web Portal & User Experience
  - Support user to add customized trial.
  - User can zoom out/in in detail graphs, except Hyper-parameter.
* Documentation
  - Improved NNI API documentation with more API docstring.

### Bug fix
  - Fix the table sort issue when failed trials haven't metrics. -Issue #1773
  - Maintain selected status(Maximal/Minimal) when the page switched. -PR#1710
  - Make hyper-parameters graph's default metric yAxis more accurate. -PR#1736
  - Fix GPU script permission issue. -Issue #1665

## Release 1.1 - 10/23/2019

### Major Features

* New tuner: [PPO Tuner](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tuner/PPOTuner.md)
* [View stopped experiments](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/Nnictl.md#view)
* Tuners can now use dedicated GPU resource (see `gpuIndices` in [tutorial](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/ExperimentConfig.md) for details)
* Web UI improvements
  - Trials detail page can now list hyperparameters of each trial, as well as their start and end time (via "add column")
  - Viewing huge experiment is now less laggy
* More examples
  - [EfficientNet PyTorch example](https://github.com/ultmaster/EfficientNet-PyTorch)
  - [Cifar10 NAS example](https://github.com/microsoft/nni/blob/v1.1/examples/trials/nas_cifar10/README.md)
* [Model compression toolkit - Alpha release](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Compressor/Overview.md): We are glad to announce the alpha release for model compression toolkit on top of NNI, it's still in the experiment phase which might evolve based on usage feedback. We'd like to invite you to use, feedback and even contribute

### Fixed Bugs

* Multiphase job hangs when search space exhuasted (issue #1204)
* `nnictl` fails when log not available (issue #1548)

## Release 1.0 - 9/2/2019

### Major Features

* Tuners and Assessors
    - Support Auto-Feature generator & selection    -Issue#877  -PR #1387
         + Provide auto feature interface
         + Tuner based on beam search
         + [Add Pakdd example](https://github.com/microsoft/nni/tree/master/examples/trials/auto-feature-engineering)
    - Add a parallel algorithm to improve the performance of TPE with large concurrency.  -PR #1052
    - Support multiphase for hyperband    -PR #1257

* Training Service
     - Support private docker registry   -PR #755

 * Engineering Improvements
    - Python wrapper for rest api, support retrieve the values of the metrics in a programmatic way  PR #1318
    - New python API : get_experiment_id(), get_trial_id()  -PR #1353   -Issue #1331 & -Issue#1368
    - Optimized NAS Searchspace  -PR #1393
         + Unify NAS search space with _type -- "mutable_type"e
         + Update random search tuner
    - Set gpuNum as optional      -Issue #1365
    - Remove outputDir and dataDir configuration in PAI mode   -Issue #1342
    - When creating a trial in Kubeflow mode, codeDir will no longer be copied to logDir   -Issue #1224

* Web Portal & User Experience
    - Show the best metric curve during search progress in WebUI  -Issue #1218
    - Show the current number of parameters list in multiphase experiment   -Issue1210  -PR #1348
    - Add "Intermediate count" option in AddColumn.      -Issue #1210
    - Support search parameters value in WebUI     -Issue #1208
    - Enable automatic scaling of axes for metric value  in default metric graph   -Issue #1360
    - Add a detailed documentation link to the nnictl command in the command prompt    -Issue #1260
    - UX improvement for showing Error log   -Issue #1173

* Documentation
    - Update the docs structure  -Issue #1231
    - (deprecated) Multi phase document improvement   -Issue #1233  -PR #1242
         + Add configuration example
    - [WebUI description improvement](Tutorial/WebUI.md)  -PR #1419


### Bug fix
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

## Release 0.9 - 7/1/2019

### Major Features
* General NAS programming interface
    * Add `enas-mode`  and `oneshot-mode` for NAS interface: [PR #1201](https://github.com/microsoft/nni/pull/1201#issue-291094510)
* [Gaussian Process Tuner with Matern kernel](Tuner/GPTuner.md)

* (deprecated) Multiphase experiment supports
    * Added new training service support for multiphase experiment: PAI mode supports multiphase experiment since v0.9.
    * Added multiphase capability for the following builtin tuners:
        * TPE, Random Search, Anneal, Naïve Evolution, SMAC, Network Morphism, Metis Tuner.

* Web Portal
    * Enable trial comparation in Web Portal. For details, refer to [View trials status](Tutorial/WebUI.md)
    * Allow users to adjust rendering interval of Web Portal. For details, refer to [View Summary Page](Tutorial/WebUI.md)
    * show intermediate results more friendly. For details, refer to [View trials status](Tutorial/WebUI.md)
* [Commandline Interface](Tutorial/Nnictl.md)
    * `nnictl experiment delete`: delete one or all experiments, it includes log, result, environment information and cache. It uses to delete useless experiment result, or save disk space.
    * `nnictl platform clean`: It uses to clean up disk on a target platform. The provided YAML file includes the information of target platform, and it follows the same schema as the NNI configuration file.
### Bug fix and other changes
* Tuner Installation Improvements: add [sklearn](https://scikit-learn.org/stable/) to nni dependencies.
* (Bug Fix) Failed to connect to PAI http code - [Issue #1076](https://github.com/microsoft/nni/issues/1076)
* (Bug Fix) Validate file name for PAI platform - [Issue #1164](https://github.com/microsoft/nni/issues/1164)
* (Bug Fix) Update GMM evaluation in Metis Tuner
* (Bug Fix) Negative time number rendering in Web Portal - [Issue #1182](https://github.com/microsoft/nni/issues/1182), [Issue #1185](https://github.com/microsoft/nni/issues/1185)
* (Bug Fix) Hyper-parameter not shown correctly in WebUI when there is only one hyper parameter - [Issue #1192](https://github.com/microsoft/nni/issues/1192)

## Release 0.8 - 6/4/2019

### Major Features

* Support NNI on Windows for OpenPAI/Remote mode
  * NNI running on windows for remote mode
  * NNI running on windows for OpenPAI mode
* Advanced features for using GPU
  * Run multiple trial jobs on the same GPU for local and remote mode
  * Run trial jobs on the GPU running non-NNI jobs
* Kubeflow v1beta2 operator
  * Support Kubeflow TFJob/PyTorchJob v1beta2
* [General NAS programming interface](https://github.com/microsoft/nni/blob/v0.8/docs/en_US/GeneralNasInterfaces.md)
  * Provide NAS programming interface for users to easily express their neural architecture search space through NNI annotation
  * Provide a new command `nnictl trial codegen` for debugging the NAS code
  * Tutorial of NAS programming interface, example of NAS on MNIST, customized random tuner for NAS
* Support resume tuner/advisor's state for experiment resume
* For experiment resume, tuner/advisor will be resumed by replaying finished trial data
* Web Portal
  * Improve the design of copying trial's parameters
  * Support 'randint' type in hyper-parameter graph
  * Use should ComponentUpdate to avoid unnecessary render

### Bug fix and other changes

* Bug fix that `nnictl update` has inconsistent command styles
* Support import data for SMAC tuner
* Bug fix that experiment state transition from ERROR back to RUNNING
* Fix bug of table entries
* Nested search space refinement
* Refine 'randint' type and support lower bound
* [Comparison of different hyper-parameter tuning algorithm](CommunitySharings/HpoComparison.md)
* [Comparison of NAS algorithm](CommunitySharings/NasComparison.md)
* [NNI practice on Recommenders](CommunitySharings/RecommendersSvd.md)

## Release 0.7 - 4/29/2018

### Major Features

* [Support NNI on Windows](Tutorial/InstallationWin.md)
  * NNI running on windows for local mode
* [New advisor: BOHB](Tuner/BohbAdvisor.md)
  * Support a new advisor BOHB, which is a robust and efficient hyperparameter tuning algorithm, combines the advantages of Bayesian optimization and Hyperband
* [Support import and export experiment data through nnictl](Tutorial/Nnictl.md)
  * Generate analysis results report after the experiment execution
  * Support import data to tuner and advisor for tuning
* [Designated gpu devices for NNI trial jobs](Tutorial/ExperimentConfig.md#localConfig)
  * Specify GPU devices for NNI trial jobs by gpuIndices configuration, if gpuIndices is set in experiment configuration file, only the specified GPU devices are used for NNI trial jobs.
* Web Portal enhancement
  * Decimal format of metrics other than default on the Web UI
  * Hints in WebUI about Multi-phase
  * Enable copy/paste for hyperparameters as python dict
  * Enable early stopped trials data for tuners.
* NNICTL provide better error message
  * nnictl provide more meaningful error message for YAML file format error

### Bug fix

* Unable to kill all python threads after nnictl stop in async dispatcher mode
* nnictl --version does not work with make dev-install
* All trail jobs status stays on 'waiting' for long time on OpenPAI platform

## Release 0.6 - 4/2/2019

### Major Features

* [Version checking](TrainingService/PaiMode.md)
  * check whether the version is consistent between nniManager and trialKeeper
* [Report final metrics for early stop job](https://github.com/microsoft/nni/issues/776)
  * If includeIntermediateResults is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result. The default value of includeIntermediateResults is false.
* [Separate Tuner/Assessor](https://github.com/microsoft/nni/issues/841)
  * Adds two pipes to separate message receiving channels for tuner and assessor.
* Make log collection feature configurable
* Add intermediate result graph for all trials

### Bug fix

* [Add shmMB config key for OpenPAI](https://github.com/microsoft/nni/issues/842)
* Fix the bug that doesn't show any result if metrics is dict
* Fix the number calculation issue for float types in hyperband
* Fix a bug in the search space conversion in SMAC tuner
* Fix the WebUI issue when parsing experiment.json with illegal format
* Fix cold start issue in Metis Tuner

## Release 0.5.2 - 3/4/2019

### Improvements

* Curve fitting assessor performance improvement.

### Documentation
* Chinese version document: https://nni.readthedocs.io/zh/latest/
* Debuggability/serviceability document: https://nni.readthedocs.io/en/latest/Tutorial/HowToDebug.html
* Tuner assessor reference: https://nni.readthedocs.io/en/latest/sdk_reference.html

### Bug Fixes and Other Changes
* Fix a race condition bug that does not store trial job cancel status correctly.
* Fix search space parsing error when using SMAC tuner.
* Fix cifar10 example broken pipe issue.
* Add unit test cases for nnimanager and local training service.
* Add integration test azure pipelines for remote machine, OpenPAI and kubeflow training services.
* Support Pylon in OpenPAI webhdfs client.

## Release 0.5.1 - 1/31/2018
### Improvements
* Making [log directory](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md) configurable
* Support [different levels of logs](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md), making it easier for debugging

### Documentation
* Reorganized documentation & New Homepage Released: https://nni.readthedocs.io/en/latest/

### Bug Fixes and Other Changes
* Fix the bug of installation in python virtualenv, and refactor the installation logic
* Fix the bug of HDFS access failure on OpenPAI mode after OpenPAI is upgraded.
* Fix the bug that sometimes in-place flushed stdout makes experiment crash

## Release 0.5.0 - 01/14/2019

### Major Features

#### New tuner and assessor supports

* Support [Metis tuner](Tuner/MetisTuner.md) as a new NNI tuner. Metis algorithm has been proofed to be well performed for **online** hyper-parameter tuning.
* Support [ENAS customized tuner](https://github.com/countif/enas_nni), a tuner contributed by github community user, is an algorithm for neural network search, it could learn neural network architecture via reinforcement learning and serve a better performance than NAS.
* Support [Curve fitting assessor](Assessor/CurvefittingAssessor.md) for early stop policy using learning curve extrapolation.
* Advanced Support of [Weight Sharing](https://github.com/microsoft/nni/blob/v0.5/docs/AdvancedNAS.md): Enable weight sharing for NAS tuners, currently through NFS.

#### Training Service Enhancement

* [FrameworkController Training service](TrainingService/FrameworkControllerMode.md): Support run experiments using frameworkcontroller on kubernetes
  * FrameworkController is a Controller on kubernetes that is general enough to run (distributed) jobs with various machine learning frameworks, such as tensorflow, pytorch, MXNet.
  * NNI provides unified and simple specification for job definition.
  * MNIST example for how to use FrameworkController.

#### User Experience improvements

* A better trial logging support for NNI experiments in OpenPAI, Kubeflow and FrameworkController mode:
  * An improved logging architecture to send stdout/stderr of trials to NNI manager via Http post. NNI manager will store trial's stdout/stderr messages in local log file.
  * Show the link for trial log file on WebUI.
* Support to show final result's all key-value pairs.

## Release 0.4.1 - 12/14/2018

### Major Features

#### New tuner supports

* Support [network morphism](Tuner/NetworkmorphismTuner.md) as a new tuner

#### Training Service improvements

* Migrate [Kubeflow training service](TrainingService/KubeflowMode.md)'s dependency from kubectl CLI to [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) client
* [Pytorch-operator](https://github.com/kubeflow/pytorch-operator) support for Kubeflow training service
* Improvement on local code files uploading to OpenPAI HDFS
* Fixed OpenPAI integration WebUI bug: WebUI doesn't show latest trial job status, which is caused by OpenPAI token expiration

#### NNICTL improvements

* Show version information both in nnictl and WebUI. You can run **nnictl -v** to show your current installed NNI version

#### WebUI improvements

* Enable modify concurrency number during experiment
* Add feedback link to NNI github 'create issue' page
* Enable customize top 10 trials regarding to metric numbers (largest or smallest)
* Enable download logs for dispatcher & nnimanager
* Enable automatic scaling of axes for metric number
* Update annotation to support displaying real choice in searchspace

### New examples

* [FashionMnist](https://github.com/microsoft/nni/tree/master/examples/trials/network_morphism), work together with network morphism tuner
* [Distributed MNIST example](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch) written in PyTorch

## Release 0.4 - 12/6/2018

### Major Features

* [Kubeflow Training service](TrainingService/KubeflowMode.md)
  * Support tf-operator
  * [Distributed trial example](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-distributed/dist_mnist.py) on Kubeflow
* [Grid search tuner](Tuner/GridsearchTuner.md)
* [Hyperband tuner](Tuner/HyperbandAdvisor.md)
* Support launch NNI experiment on MAC
* WebUI
  * UI support for hyperband tuner
  * Remove tensorboard button
  * Show experiment error message
  * Show line numbers in search space and trial profile
  * Support search a specific trial by trial number
  * Show trial's hdfsLogPath
  * Download experiment parameters

### Others

* Asynchronous dispatcher
* Docker file update, add pytorch library
* Refactor 'nnictl stop' process, send SIGTERM to nni manager process, rather than calling stop Rest API.
* OpenPAI training service bug fix
  * Support NNI Manager IP configuration(nniManagerIp) in OpenPAI cluster config file, to fix the issue that user’s machine has no eth0 device
  * File number in codeDir is capped to 1000 now, to avoid user mistakenly fill root dir for codeDir
  * Don’t print useless ‘metrics is empty’ log in OpenPAI job’s stdout. Only print useful message once new metrics are recorded, to reduce confusion when user checks OpenPAI trial’s output for debugging purpose
  * Add timestamp at the beginning of each log entry in trial keeper.

## Release 0.3.0 - 11/2/2018

### NNICTL new features and updates

* Support running multiple experiments simultaneously.

  Before v0.3, NNI only supports running single experiment once a time. After this release, users are able to run multiple experiments simultaneously. Each experiment will require a unique port, the 1st experiment will be set to the default port as previous versions. You can specify a unique port for the rest experiments as below:

  ```bash
  nnictl create --port 8081 --config <config file path>
  ```

* Support updating max trial number.
  use `nnictl update --help` to learn more. Or refer to [NNICTL Spec](Tutorial/Nnictl.md) for the fully usage of NNICTL.

### API new features and updates

* <span style="color:red">**breaking change**</span>: nn.get_parameters() is refactored to nni.get_next_parameter. All examples of prior releases can not run on v0.3, please clone nni repo to get new examples. If you had applied NNI to your own codes, please update the API accordingly.

* New API **nni.get_sequence_id()**.
  Each trial job is allocated a unique sequence number, which can be retrieved by nni.get_sequence_id() API.

  ```bash
  git clone -b v0.3 https://github.com/microsoft/nni.git
  ```

* **nni.report_final_result(result)** API supports more data types for result parameter.

  It can be of following types:
  * int
  * float
  * A python dict containing 'default' key, the value of 'default' key should be of type int or float. The dict can contain any other key value pairs.

### New tuner support

* **Batch Tuner** which iterates all parameter combination, can be used to submit batch trial jobs.

### New examples

* A NNI Docker image for public usage:

  ```bash
  docker pull msranni/nni:latest
  ```

* New trial example: [NNI Sklearn Example](https://github.com/microsoft/nni/tree/master/examples/trials/sklearn)
* New competition example: [Kaggle Competition TGS Salt Example](https://github.com/microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### Others

* UI refactoring, refer to [WebUI doc](Tutorial/WebUI.md) for how to work with the new UI.
* Continuous Integration: NNI had switched to Azure pipelines

## Release 0.2.0 - 9/29/2018

### Major Features

* Support [OpenPAI](https://github.com/microsoft/pai) Training Platform (See [here](TrainingService/PaiMode.md) for instructions about how to submit NNI job in pai mode)
  * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
  * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
* Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](Tuner/SmacTuner.md) for instructions about how to use SMAC tuner)
  * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO to handle categorical parameters. The SMAC supported by NNI is a wrapper on [SMAC3](https://github.com/automl/SMAC3)
* Support NNI installation on [conda](https://conda.io/docs/index.html) and python virtual environment
* Others
  * Update ga squad example and related documentation
  * WebUI UX small enhancement and bug fix


## Release 0.1.0 - 9/10/2018 (initial release)

Initial release of Neural Network Intelligence (NNI).

### Major Features

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
  * Support CI by providing out-of-box integration with [travis-ci](https://github.com/travis-ci) on ubuntu
* Others
  * Support simple GPU job scheduling
