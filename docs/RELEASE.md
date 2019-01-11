# Release 0.4 - 12/6/2018

## Major Features
  * [Kubeflow Training service](./KubeflowMode.md)
    * Support tf-operator
    * [Distributed trial example](../examples/trials/mnist-distributed/dist_mnist.py) on Kubeflow
  * [Grid search tuner](../src/sdk/pynni/nni/README.md#Grid) 
  * [Hyperband tuner](../src/sdk/pynni/nni/README.md#Hyperband)
  * Support launch NNI experiment on MAC
  * WebUI 
    *  UI support for hyperband tuner
    *  Remove tensorboard button 
    *  Show experiment error message 
    *  Show line numbers in search space and trial profile
    *  Support search a specific trial by trial number
    *  Show trial's hdfsLogPath
    *  Download experiment parameters
## Others
  * Asynchronous dispatcher
  * Docker file update, add pytorch library 
  * Refactor 'nnictl stop' process, send SIGTERM to nni manager process, rather than calling stop Rest API. 
  * OpenPAI training service bug fix
    *  Support NNI Manager IP configuration(nniManagerIp) in PAI cluster config file, to fix the issue that user’s machine has no eth0 device 
    *  File number in codeDir is capped to 1000 now, to avoid user mistakenly fill root dir for codeDir
    *  Don’t print useless ‘metrics is empty’ log int PAI job’s stdout. Only print useful message once new metrics are recorded, to reduce confusion when user checks PAI trial’s output for debugging purpose
    *  Add timestamp at the beginning of each log entry in trial keeper.

# Release 0.3.0 - 11/2/2018
## NNICTL new features and updates
* Support running multiple experiments simultaneously. 

    Before v0.3, NNI only supports running single experiment once a time. After this realse, users are able to run multiple experiments simultaneously. Each experiment will require a unique port, the 1st experiment will be set to the default port as previous versions. You can specify a unique port for the rest experiments as below:

    ```nnictl create --port 8081 --config <config file path>```
* Support updating max trial number.
    use ```nnictl update --help``` to learn more. Or refer to [NNICTL Spec](https://github.com/Microsoft/nni/blob/master/docs/NNICTLDOC.md) for the fully usage of NNICTL.

## API new features and updates
* <span style="color:red">**breaking change**</span>: nn.get_parameters() is refactored to nni.get_next_parameter. All examples of prior releases can not run on v0.3, please clone nni repo to get new examples. If you had applied NNI to your own codes, please update the API accordingly.

* New API **nni.get_sequence_id()**. 
    Each trial job is allocated a unique sequence number, which can be retrieved by nni.get_sequence_id() API.

    ```git clone -b v0.3 https://github.com/Microsoft/nni.git```
* **nni.report_final_result(result)** API supports more data types for result parameter. 
    It can be of following types:
    * int
    * float
    * A python dict containing 'default' key, the value of 'default' key should be of type int or float. The dict can contain any other key value pairs.

## New tuner support
* **Batch Tuner** which iterates all parameter combination, can be used to submit batch trial jobs.

## New examples
* A NNI Docker image for public usage:
      ```docker pull msranni/nni:latest```
* New trial example: [NNI Sklearn Example](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)
* New competition example: [Kaggle Competition TGS Salt Example](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)
    
## Others
* UI refactoring, refer to [WebUI doc](WebUI.md) for how to work with the new UI.
* Continuous Integration: NNI had switched to Azure pipelines
* [Known Issues in release 0.3.0](https://github.com/Microsoft/nni/labels/nni030knownissues).


# Release 0.2.0 - 9/29/2018
## Major Features
   * Support [OpenPAI](https://github.com/Microsoft/pai) (aka pai) Training Service (See [here](./PAIMode.md) for instructions about how to submit NNI job in pai mode)
      * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
      * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
   * Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](HowToChooseTuner.md) for instructions about how to use SMAC tuner)
      * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO to handle categorical parameters. The SMAC supported by NNI is a wrapper on [SMAC3](https://github.com/automl/SMAC3)
   * Support NNI installation on [conda](https://conda.io/docs/index.html) and python virtual environment
   * Others
      * Update ga squad example and related documentation
      * WebUI UX small enhancement and bug fix

## Known Issues
[Known Issues in release 0.2.0](https://github.com/Microsoft/nni/labels/nni020knownissues).

# Release 0.1.0 - 9/10/2018 (initial release)

Initial release of Neural Network Intelligence (NNI).

## Major Features
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

## Known Issues
[Known Issues in release 0.1.0](https://github.com/Microsoft/nni/labels/nni010knownissues).
