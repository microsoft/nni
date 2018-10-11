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
      
   
