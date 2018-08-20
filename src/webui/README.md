# webui

NNI is a research platform for metalearning. It provides easy-to-use interface so that you could perform neural architecture search, hyperparameter optimization and optimizer design for your own problems and models.
Web UI allows user to monitor the status of the NNI system using a graphical interface.

## Deployment

### To start the webui

> $ yarn
> $ yarn start

## Usage

### View summary page

Click the tab "Overview".

* See good performance trial.
* See search_space json.
* See complete trial cdf graph.

### View job accuracy

Click the tab "Optimization Progress" to see the point graph of all trials. Hover every point to see its specific accuracy.

### View hyper parameter

Click the tab "Hyper Parameter" to see the parallel graph.

* You can select the percentage to cut down some lines.
* Choose two axes to swap its positions

### View trial status 

Click the tab "Trial Status" to see the status of the all trials. Specifically:

* Running trial: running trial's duration in the bar graph.
* Trial detail: trial's id, trial's duration, start time, end time, status and accuracy.
* Kill: you can kill a job that status is running.
* Tensor: you can see a job in the tensorflow graph, it will link to the Tensorboard page.
* Log: click the button, you can see the log about NNI and pai.

### Control 

Click the tab "Control" to add a new trial or update the search_space file.

### View Tensorboard Graph 
   
Click the tab "Tensorboard" to see a job in the tensorflow graph. 