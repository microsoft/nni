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
      * Provide a web UI for viewing experiments details and managing experiments
   * Continuous Integration
      * Support CI by providing out-of-box integration with [travis-ci](https://github.com/travis-ci) on ubuntu    
   * Others
      * Support simple GPU job scheduling 

## Known Issues
[Known Issues in release 0.1.0](https://github.com/Microsoft/nni/labels/nni010knownissues).
      
   
