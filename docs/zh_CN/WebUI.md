# Web 界面

## 查看概要页面

点击标签 "Overview"。

* 查看 Experiment 的配置和搜索空间内容。
* 支持下载 Experiment 结果。

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

## 查看 Trial 状态

点击 "Trials Detail" 标签查看所有 Trial 的状态。 特别是：

* Trial 详情：Trial 的 id，持续时间，开始时间，结束时间，状态，精度和搜索空间。

![](../img/webui-img/detail-local.png)

* 如果在 OpenPAI 或 Kubeflow 平台上运行，还可以看到 hdfsLog。

![](../img/webui-img/detail-pai.png)

* Kill: 可终止正在运行的任务。
* 支持搜索某个特定的 Trial。
* 中间结果图。

![](../img/intermediate.png)