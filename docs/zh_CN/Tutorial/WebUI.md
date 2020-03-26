# Web 界面

## 查看概要页面

点击标签 "Overview"。

* 在 Overview 标签上，可看到 Experiment Trial 的概况、搜索空间、以及最好的 Trial 结果。

![](../../img/webui-img/over1.png) ![](../../img/webui-img/over2.png)

* 如果 Experiment 包含了较多 Trial，可改变刷新间隔。

![](../../img/webui-img/refresh-interval.png)

* "View" 按钮支持查看并下载 Experiment 结果，以及 NNI Manager、Dispatcher 的日志文件。

![](../../img/webui-img/download.png)

* 如果实验的状态为错误，可以单击错误框中的感叹号来查看日志消息。

![](../../img/webui-img/log-error.png) ![](../../img/webui-img/review-log.png)

* You can click "Feedback" to report any questions.

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

Trial 可能在训练过程中有大量中间结果。 为了更清楚的理解一些 Trial 的趋势，可以为中间结果图设置过滤。

这样可以发现 Trial 在某个中间结果上会变得更好或更差。 This indicates that it is an important and relevant intermediate result. To take a closer look at the point here, you need to enter its corresponding X-value at #Intermediate. Then input the range of metrics on this intermedia result. In the picture below, we choose the No. 4 intermediate result and set the range of metrics to 0.8-1.

![](../../img/webui-img/filter-intermediate.png)

## 查看 Trial 状态

Click the tab "Trials Detail" to see the status of all trials. Specifically:

* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy, and search space file.

![](../../img/webui-img/detail-local.png)

* The button named "Add column" can select which column to show on the table. If you run an experiment whose final result is a dict, you can see other keys in the table. 可选择 "Intermediate count" 列来查看 Trial 进度。

![](../../img/webui-img/addColumn.png)

* 如果要比较某些 Trial，可选择并点击 "Compare" 来查看结果。

![](../../img/webui-img/select-trial.png) ![](../../img/webui-img/compare.png)

* 支持通过 id，状态，Trial 编号， 以及参数来搜索。

![](../../img/webui-img/search-trial.png)

* You can use the button named "Copy as python" to copy the trial's parameters.

![](../../img/webui-img/copyParameter.png)

* If you run on the OpenPAI or Kubeflow platform, you can also see the hdfsLog.

![](../../img/webui-img/detail-pai.png)

* Intermediate Result Graph: you can see the default and other keys in this graph by clicking the operation column button.

![](../../img/webui-img/intermediate-btn.png) ![](../../img/webui-img/intermediate.png)

* Kill: 可终止正在运行的任务。

![](../../img/webui-img/kill-running.png) ![](../../img/webui-img/canceled.png)