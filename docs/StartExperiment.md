How to start an experiment
===
## 1.Introduce
There are few steps to start an new experiment of nni, here are the  process.
<img src="./img/experiment_process.jpg" width="50%" height="50%" />
## 2.Details
### 2.1 Check environment
The first step to start an experiment is to check whether the environment is ready, nnictl will check if there is an old experiment running or the port of restfurl server is occupied.
NNICTL will also validate the content of config yaml file, to ensure the experiment config is in correct format.

### 2.2 Check restful server
After check environment, nnictl will start an restful server process to manage nni experiment, the devault port is 51188.

### 2.3 Check restful server
Before next steps, nnictl will check whether restful server is successfully started, or the starting process will stop and show error message.

### 2.4 Set experiment config
NNICTL need to set experiment config before start an experiment, experiment config includes the config values in config yaml file.

### 2.5 Check experiment cofig
NNICTL will ensure the request to set config is successfully executed.

### 2.6 Start Web UI
NNICTL will start a Web UI process to show Web UI information,the default port of Web UI is 8080.

### 2.7 Check Web UI
If Web UI is not successfully started, nnictl will give a warning information, and will continue to start experiment.

### 2.8 Start Experiment
This is the most import step of starting an nni experiment, nnictl will call restful server process to setup an experiment.

### 2.9 Check experiment
After start experiment, nnictl will check whether the experiment is correctly created, and show more information of this experiment to users.