# Web 界面

## 查看概要页面

点击标签 "Overview"。

* 查看实验参数。
* 查看最好结果的尝试。
* 查看搜索空间 JSON 文件。

## 查看任务准确度

Click the tab "Optimization Progress" to see the point graph of all trials. 将鼠标悬停到某个点查看其准确度。

## 查看超参

点击 "Hyper Parameter" 标签查看图像。

* 可选择百分比查看最好的尝试。
* 选择两个轴来交换位置。

## 查看尝试状态

Click the tab "Trial Status" to see the status of the all trials. 特别是：

* Trial duration：尝试执行时间的条形图。
* 尝试详情：尝试的 id，持续时间，开始时间，结束时间，状态，精度和搜索空间。
* Kill: 可终止正在运行的任务。
* Tensor: you can see a job in the tensorflow graph, it will link to the Tensorboard page.

## Control

Click the tab "Control" to add a new trial or update the search_space file and some experiment parameters.

## Feedback

[Known Issues](https://github.com/Microsoft/nni/issues).