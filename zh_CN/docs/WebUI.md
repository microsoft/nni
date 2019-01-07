# WebUI

## 查看概要页面

Click the tab "Overview".

* See the experiment trial profile and search space message.
* Support to download the experiment result.

![](../../docs/img/over1.png)

* See good performance trials.

![](../../docs/img/over2.png)

## View job default metric

Click the tab "Default Metric" to see the point graph of all trials. Hover to see its specific default metric and search space message.

![](../../docs/img/accuracy.png)

## View hyper parameter

Click the tab "Hyper Parameter" to see the parallel graph.

* You can select the percentage to see top trials.
* Choose two axis to swap its positions

![](../../docs/img/hyperPara.png)

## View Trial Duration

Click the tab "Trial Duration" to see the bar graph.

![](../../docs/img/trial_duration.png)

## View trials status

Click the tab "Trials Detail" to see the status of the all trials. Specifically:

* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy and search space file.
* If you run a pai experiment, you can also see the hdfsLogPath.

![](../../docs/img/table_openrow.png)

* Kill: you can kill a job that status is running.
* Support to search for a specific trial.
* Intermediate Result Graph.

![](../../docs/img/intermediate.png)