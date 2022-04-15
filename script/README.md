
## Overview
This branch is for the OSDI'22 artifact evaluation of paper "SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute". 


## Evaluation Setup
* Artifacts Available:
The source code of Sparta is available at: https://github.com/microsoft/nni

* Artifacts Functional:
Documentation: the following document includes detailed guidelines on how to build, install, test Sparta and the experiments to compare with other baselines.

* Results Reproduced:
To reproduce the main results presented in our paper, we provide a Docker image containing all the environments and baseline softwares. We also provide detailed guideline to help reproduce the results step by step.


## Environment setup
First, git clone the source code.
```
git clone https://github.com/microsoft/nni
cd nni && git checkout sparta_artifact
```
To make the reproducing easier, we provide a docker image that contains all dependencies and baselines. Build the docker image:
```
cd image
sudo docker build . -t artifact
```
Third, start a docker instance
```
sudo docker run -it --gpus all --shm-size 16G artifact
```
Following commands are executed in the docker.
First we also need get the source code and initialize the environment
```

# install the sparta
git clone https://github.com/microsoft/nni && cd nni && git checkout sparta_artifact
conda activate artifact
cd SparTA && python setup.py develop
# initialize the environment
cd script && bash init_env.sh
```
## Run the experiments
### 2080Ti
We provide an end-to-end script to reproduce the experiment results of 2080Ti and A100. To reproduce the 2080Ti experiments with one command:
```
bash run_all_2080ti.sh # current directory should be $SPARTA_HOME/script
```
If you want to reproduce the figure in the paper one by one, you can first generate the checkpoints used in the following experiment and then go into the corresponding directory and execute `bash run.sh`
For example:
```
bash init_checkpints.sh
# take figure13 for example
cd figure13 && bash run.sh
```
The script will produce and draw the experimental data automatically. The generated pdfs are saved into corresponding directories.


### A100
To reproduce the experiments on A100, we need to build a new docker image. The docker file of A100 environment is at `figure16/Dockerfile`. Once we build the docker image successfully, we
can also run the experiment with one script.
```
bash run_all_a100.sh # current directory should be $SPARTA_HOME/script
```
