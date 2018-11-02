# NNI Road Map
The following is a summary of the NNI team's backlog for the next 6 months. Some completed items (either in a release or master branch) are included to provide the context and progress of the work.
* New Tuners Supports
    * New MSRA research Algorithm
    * Hyperband
    * Grid search
    * Search Algorithm based on Network Morphism
* New Training Services Supports
    * Kubernetes
* User Interface and Experiences 
    * Support specifying neural architecture's search space through nni APIs in an easy way. 
    * Web UX refactor
* Support Mac and Windows
* VSCode NNI plugin
* Support more high level interface (e.g., nni.fit(), nni.predict(), similar to auto-sklearn) for users to simply feed data and then get the best model
* Support more efficient trial job training by leveraging optimizations in system level 
* Support automatic model selection and serving
* Support tensorflow.js
* Enhance debug ability for training scripts. for example to run a trial directly to test code
* Support Ensemble solution

# Release 0.3.0 - 11/2/2018
## Major Features
* Support running multiple experiments simultaneously. You can run multiple experiments by specifying a unique port for each experiment:

    ```nnictl create --port 8081 --config <config file path>```

    You can still run the first experiment without '--port' parameter:

    ```nnictl create --config <config file path>```
* A builtin Batch Tuner which iterates all parameter combination, can be used to submit batch trial jobs.
* nni.report_final_result(result) API supports more data types for result parameter, it can be of following types:
    * int
    * float
    * A python dict containing 'default' key, the value of 'default' key should be of type int or float. The dict can contain any other key value pairs.
* Continuous Integration
    * Switched to Azure pipelines
* Others
    * New nni.get_sequence_id() API. Each trial job is allocated a unique sequence number, which can be retrieved by nni.get_sequence_id() API.
    * Download experiment result from WebUI
    * Add trial examples using sklearn and NNI together
    * Support updating max trial number
    * Kaggle competition TGS Salt code as an example
    * NNI Docker image:
      ```
      docker pull msranni/nni:latest
      ```
## Breaking changes
*   <span style="color:red">API nn.get_parameters() is renamed to nni.get_next_parameter(), this is a broken change, all examples of prior releases can not run on v0.3, please clone nni repo to get new examples.</span>

    ```git clone -b v0.3 https://github.com/Microsoft/nni.git```

## Know issues
[Known Issues in release 0.3.0](https://github.com/Microsoft/nni/labels/nni030knownissues).

# Release 0.2.0 - 9/29/2018
## Major Features
   * Support [OpenPAI](https://github.com/Microsoft/pai) (aka pai) Training Service (See [here](./PAIMode.md) for instructions about how to submit NNI job in pai mode)
      * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
      * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
   * Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](../src/sdk/pynni/nni/README.md) for instructions about how to use SMAC tuner)
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
   * Tuners, Accessors and Trial
      * Support AutoML algorithms including:  hyperopt_tpe, hyperopt_annealing, hyperopt_random, and evolution_tuner
      * Support assessor(early stop) algorithms including: medianstop algorithm
      * Provide Python API for user defined tuners and accessors
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
      
   
