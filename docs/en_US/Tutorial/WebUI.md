# WebUI

## View summary page

Click the tab "Overview".

* See the experiment trial profile/search space and performanced good trials.

![](../../img/webui-img/over1.png)
![](../../img/webui-img/over2.png)
* If your experiment have many trials, you can change the refresh interval on here.

![](../../img/webui-img/refresh-interval.png)
* Support to review and download the experiment result and nni-manager/dispatcher log file from the "View" button.

![](../../img/webui-img/download.png)
* You can click the learn about in the error box to track experiment log message if the experiment's status is error.

![](../../img/webui-img/log-error.png)
![](../../img/webui-img/review-log.png)

* You can click "Feedback" to report it if you have any questions.

## View job default metric

* Click the tab "Default Metric" to see the point graph of all trials. Hover to see its specific default metric and search space message.

![](../../img/webui-img/default-metric.png)

* Click the switch named "optimization curve" to see the experiment's optimization curve.

![](../../img/webui-img/best-curve.png)

## View hyper parameter

Click the tab "Hyper Parameter" to see the parallel graph.

* You can select the percentage to see top trials.
* Choose two axis to swap its positions

![](../../img/hyperPara.png)
## View Trial Duration

Click the tab "Trial Duration" to see the bar graph.

![](../../img/trial_duration.png)
## View Trial Intermediate Result Graph

Click the tab "Intermediate Result" to see the lines graph.

![](../../img/webui-img/trials_intermeidate.png)

The trial may have many intermediate results in the training progress. In order to see the trend of some trials more clearly, we set a filtering function for the intermediate result graph.

You may find that these trials will get better or worse at one of intermediate results. In other words, this is an important and relevant intermediate result. To take a closer look at the point here, you need to enter its corresponding abscissa value at #Intermediate.

And then input the range of metrics on this intermedia result. Like below picture, it chooses No. 4 intermediate result and set the range of metrics to 0.8-1.

![](../../img/webui-img/filter-intermediate.png)
## View trials status

Click the tab "Trials Detail" to see the status of the all trials. Specifically:

* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy and search space file.

![](../../img/webui-img/detail-local.png)
* The button named "Add column" can select which column to show in the table. If you run an experiment that final result is dict, you can see other keys in the table. You can choose the column "Intermediate count" to watch the trial's progress.

![](../../img/webui-img/addColumn.png)
* If you want to compare some trials, you can select them and then click "Compare" to see the results.

![](../../img/webui-img/select-trial.png)
![](../../img/webui-img/compare.png)
* Support to search for a specific trial by it's id, status, Trial No. and parameters.

![](../../img/webui-img/search-trial.png)
* You can use the button named "Copy as python" to copy trial's parameters.

![](../../img/webui-img/copyParameter.png)
* If you run on OpenPAI or Kubeflow platform, you can also see the hdfsLog.

![](../../img/webui-img/detail-pai.png)
* Intermediate Result Graph: you can see default and other keys in this graph by click the operation column button.

![](../../img/webui-img/intermediate-btn.png)
![](../../img/webui-img/intermediate.png)
* Kill: you can kill a job that status is running.

![](../../img/webui-img/kill-running.png)
![](../../img/webui-img/canceled.png)
