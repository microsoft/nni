# Web 界面

## 查看概要页面

点击标签 "Overview"。

* 查看 Experiment 的配置和搜索空间内容。
* 支持下载 Experiment 结果。
* Support to export nni-manager and dispatcher log file.
* If you have any question, you can click "Feedback" to report it.

![](../img/webui-img/over1.png)

* 查看最好结果的 Trial。

![](../img/webui-img/over2.png)

## 查看任务默认指标

点击 "Default Metric" 来查看所有 Trial 的点图。 悬停鼠标来查看默认指标和搜索空间信息。

![](../img/accuracy.png)

## 查看超参

点击 "Hyper Parameter" 标签查看图像。

* 可选择百分比查看最好的 Trial。
* 选择两个轴来交换位置。

![](../img/hyperPara.png)

## 查看 Trial 运行时间

点击 "Trial Duration" 标签来查看柱状图。

![](../img/trial_duration.png)

## 查看 Trial 中间结果

单击 "Intermediate Result" 标签查看折线图。

![](../img/webui-img/trials_intermeidate.png)

该图有筛选功能。 点击筛选按钮， 在第一个输入框中输入关注点的序号， 在第二个输入框中输入中间结果的范围，选出需要的数据。

![](../img/webui-img/filter_intermediate.png)

## 查看 Trial 状态

点击 "Trials Detail" 标签查看所有 Trial 的状态。 特别是：

* Trial 详情：Trial 的 id，持续时间，开始时间，结束时间，状态，精度和搜索空间。

![](../img/webui-img/detail-local.png)

* The button named "Add column" can select which column to show in the table. If you run an experiment that final result is dict, you can see other keys in the table.

![](../img/webui-img/addColumn.png)

* You can use the button named "Copy as python" to copy trial's parameters.

![](../img/webui-img/copyParameter.png)

* If you run on OpenPAI or Kubeflow platform, you can also see the hdfsLog.

![](../img/webui-img/detail-pai.png)

* Kill: you can kill a job that status is running.
* Support to search for a specific trial.
* Intermediate Result Graph.

![](../img/intermediate.png)