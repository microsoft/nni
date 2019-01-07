# WebUI

## 查看概要页面

点击标签 "Overview"。

* 查看实验的配置和搜索空间内容。
* 支持下载实验结果。

![](../../docs/img/over1.png)

* 查看最好结果的尝试。

![](../../docs/img/over2.png)

## 查看任务默认指标

点击 "Default Metric" 来查看所有尝试的点图。 悬停鼠标来查看默认指标和搜索空间信息。

![](../../docs/img/accuracy.png)

## 查看超参

点击 "Hyper Parameter" 标签查看图像。

* 可选择百分比查看最好的尝试。
* 选择两个轴来交换位置。

![](../../docs/img/hyperPara.png)

## 查看尝试运行时间

点击 "Trial Duration" 标签来查看柱状图。

![](../../docs/img/trial_duration.png)

## 查看尝试状态

Click the tab "Trials Detail" to see the status of the all trials. Specifically:

* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy and search space file.
* If you run a pai experiment, you can also see the hdfsLogPath.

![](../../docs/img/table_openrow.png)

* Kill: you can kill a job that status is running.
* Support to search for a specific trial.
* Intermediate Result Graph.

![](../../docs/img/intermediate.png)