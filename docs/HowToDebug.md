**How to Debug in NNI**
===

## Overview

There are three parts that might have logs in NNI. They are nnimanager, dispatcher and trial. Here we will introduce them succinctly. More information please refer to [Overview](Overview.md).

- **nnimanager**: nnimanager is the core of NNI, whose log is important when the whole experiment fails (e.g., no webUI or training service fails)
- **Dispatcher**: Dispatcher is the collective name of **Tuner** and **Assessor**. Logs of dispatcher are related to the tuner or assessor code.
    - **Tuner**: Tuner is an AutoML algorithm, which generates a new configuration for the next try. A new trial will run with this configuration.
    - **Assessor**: Assessor analyzes trial's intermediate results (e.g., periodically evaluated accuracy on test dataset) to tell whether this trial can be early stopped or not.
- **Trial**: Trial code is the code you write to run your experiment, which is an individual attempt at applying a new configuration (e.g., a set of hyperparameter values, a specific nerual architecture).

## Where is the log

### Experiment Root Directory
Every experiment has a root folder, which is shown on the right-top corner of webUI. Or you could assemble it by replacing the `experiment_id` with your actual experiment_id in path `~/nni/experiment/experiment_id/` in case of webUI failure. `experiment_id` could be seen when you run `nnictl create ...` to create a new experiment.

Under the root path, there is a directory named `log`, where `nnimanager.log` and `dispatcher.log` are there.

### Trial Root Directory

Usually in webUI, you can click `+` in the left of every trial to expand it to see each trial's log path. Also there is another directory under experiment root directory, named `trials-local`, which stores all the trials run. Every trial has a unique id that is the name of its root directory. In this directory, a file named `stderr` records trial error and another named `trial.log` records this trial's log. 

## Different kinds of errors

There are different kinds of errors. However, they can be divided into three categories based on their severity. So when nni fails, check each part sequentially.

### **nnimanger** Failed

Usually this is the most serious error. When this happens, the whole experiment fails and no trial will be run.

When this happens, you should check the nnimanager's log to find if there is any error.




### **Dispatcher** Failed

Dispatcher fails. Usually for some new users of NNI, it means that tuner fails. You could check dispatcher's log to see what happens to your dispatcher. For built-in tuner, some common errors might be invalid search space (unsupported type of search space).


### **Trial** Failed

In this situation, NNI can still run and dispatch trials. 

It means your trial code (which is run by NNI) fails. This kind of error is strongly related to your trial code. Please check trial's log to fix any possible errors shown there.

A common example of this would be run the mnist example without installing tensorflow. Surely there is an Import Error in your trial code and thus every trial fails

