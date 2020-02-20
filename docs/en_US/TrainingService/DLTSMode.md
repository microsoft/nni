**Run an Experiment on Deep Learning Training Service**
===
NNI supports running an experiment on [Deep Learning Training Service](https://github.com/microsoft/DLWorkspace.git) (aka DLTS), called dlts mode. Before starting to use NNI dlts mode, you should have an account to access DLTS dashboard.

## Setup Environment

Step 1. Choose a cluster from DLTS dashboard, ask administrator for the cluster dashboard URL.

![Choose Cluster](../../img/dlts-step1.png)

Step 2. Prepare a NNI config YAML like the following:

```yaml
trainingServicePlatform: dlts
authorName: your_name
experimentName: auto_mnist
# how many trials could be concurrently running
trialConcurrency: 2
# maximum experiment running duration
maxExecDuration: 3h
# empty means never stop
maxTrialNum: 100
# search space file
searchSpacePath: search_space.json
# choice: true, false
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 1
  image: msranni/nni
dltsConfig:
  dashboard: # Ask administrator for the cluster dashboard URL
```

Remember to fill the cluster dashboard URL to the last line.

Step 3. Open your working directory of the cluster, paste the NNI config as well as related code to a directory.

![Copy Config](../../img/dlts-step3.png)

Step 4. Submit a NNI manager job to the specified cluster.

![Submit Job](../../img/dlts-step4.png)

Step 5. Go to Endpoints tab of the newly created job, click the Port 40000 link to theck trial's information.

![View NNI WebUI](../../img/dlts-step5.png)
