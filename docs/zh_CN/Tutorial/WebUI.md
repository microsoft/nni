# Web 界面

## 查看概要页面

点击标签 "Overview"。

* 查看 Experiment Trial 配置、搜索空间以及结果好的 Trial。

![](../../img/webui-img/over1.png) ![](../../img/webui-img/over2.png)

* 如果 Experiment 包含了较多 Trial，可改变刷新间隔。

![](../../img/webui-img/refresh-interval.png)

* 支持查看并下载 Experiment 结果，以及 NNI Manager、Dispatcher 的日志文件。

![](../../img/webui-img/download.png)

* 如果 Experiment 状态为 ERROR，可点击图标，查看 Experiment 错误日志。

![](../../img/webui-img/log-error.png) ![](../../img/webui-img/review-log.png)

* 点击 "Feedback" 反馈问题。

## 查看任务默认指标

* 点击 "Default Metric" 来查看所有 Trial 的点图。 悬停鼠标来查看默认指标和搜索空间信息。

![](../../img/webui-img/default-metric.png)

* 点击开关 "optimization curve" 来查看 Experiment 的优化曲线。

![](../../img/webui-img/best-curve.png)

## 查看超参

点击 "Hyper Parameter" 标签查看图像。

* 可选择百分比查看最好的 Trial。
* 选择两个轴来交换位置。

![](../../img/hyperPara.png)

## 查看 Trial 运行时间

点击 "Trial Duration" 标签来查看柱状图。

![](../../img/trial_duration.png)

## 查看 Trial 中间结果

单击 "Intermediate Result" 标签查看折线图。

![](../../img/webui-img/trials_intermeidate.png)

We set a filter function for the intermediate result graph because that the trials may have many intermediate results in the training progress. You need to provide data if you want to use the filter button to see the trend of some trial.

What data should be written in the first input? Maybe you find an intermediate count those trials became better or worse. In other word, it's an important and concerned intermediate count. Just input it into the first input.

After selecting the intermeidate count, you should input your focus metric's range on this intermediate count. Yes, it's the min and max value. Like this picture, I choose the intermeidate count is 9 and the metric's range is 60-80.

As a result, I filter these trials that the metric's range is 20-60 on the 13 intermediate count.

![](../../img/webui-img/filter-intermediate.png)

## 查看 Trial 状态

Click the tab "Trials Detail" to see the status of the all trials. Specifically:

* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy and search space file.

![](../../img/webui-img/detail-local.png)

* The button named "Add column" can select which column to show in the table. If you run an experiment that final result is dict, you can see other keys in the table. You can choose the column "Intermediate count" to watch the trial's progress.

![](../../img/webui-img/addColumn.png)

* If you want to compare some trials, you can select them and then click "Compare" to see the results.

![](../../img/webui-img/select-trial.png) ![](../../img/webui-img/compare.png)

* Support to search for a specific trial by it's id, status, Trial No. and parameters.

![](../../img/webui-img/search-trial.png)

* You can use the button named "Copy as python" to copy trial's parameters.

![](../../img/webui-img/copyParameter.png)

* If you run on OpenPAI or Kubeflow platform, you can also see the hdfsLog.

![](../../img/webui-img/detail-pai.png)

* Intermediate Result Graph: you can see default and other keys in this graph by click the operation column button.

![](../../img/webui-img/intermediate-btn.png) ![](../../img/webui-img/intermediate.png)

* Kill: you can kill a job that status is running.

![](../../img/webui-img/kill-running.png) ![](../../img/webui-img/canceled.png)