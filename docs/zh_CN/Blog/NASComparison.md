# NAS 算法对比

*匿名作者*

对比 Autokeras, DARTS, ENAS 和 NAO 这些算法的效果。

源码链接如下：

- Autokeras: <https://github.com/jhfjhfj1/autokeras>

- DARTS: <https://github.com/quark0/darts>

- ENAS: <https://github.com/melodyguan/enas>

- NAO: <https://github.com/renqianluo/NAO>

## 实验描述

为了避免算法仅仅在** CIFAR-10** 这一个数据集合上过拟合，我们还对比了包括 Fashion-MNIST, CIFAR-100, OUI-Adience-Age, ImageNet-10-1 (ImageNet的一个子集) 和 ImageNet-10-2 (ImageNet的另一个子集) 在内的其他 5 种数据集。 我们分别从ImageNet中抽取10种不同类别标签的子集，组成 ImageNet10-1 和 ImageNet10-2数据集 。

| 数据集                                                                                     | 训练数据集大小 | 类别标签数 | 数据集说明                                                      |
|:--------------------------------------------------------------------------------------- | ------- | ----- | ---------------------------------------------------------- |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)                       | 60,000  | 10    | T恤上衣，裤子，套头衫，连衣裙，外套，凉鞋，衬衫，运动鞋，包和踝靴。                         |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                                 | 50,000  | 10    | 飞机，汽车，鸟类，猫，鹿，狗，青蛙，马，船和卡车。                                  |
| [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)                                | 50,000  | 100   | 和 CIFAR-10 类似，不过总共有100个类，每个类有600张图。                        |
| [OUI-Adience-Age](https://talhassner.github.io/home/projects/Adience/Adience-data.html) | 26,580  | 8     | 8个年龄组类别 (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-). |
| [ImageNet-10-1](http://www.image-net.org/)                                              | 9,750   | 10    | 咖啡杯、电脑键盘、餐桌、衣柜、割草机、麦克风、秋千、缝纫机、里程表和燃气泵。                     |
| [ImageNet-10-2](http://www.image-net.org/)                                              | 9,750   | 10    | 鼓，班吉，口哨，三角钢琴，小提琴，管风琴，原声吉他，长号，长笛和萨克斯。                       |

我们没有改变他们源码种的Fine-tuning 方法。 为了匹配每个任务，我们改变源码中模型的输入图片大小和输出类别数目的部分。

所有 NAS 方法模型搜索时间和重训练时间都是**两天**。 所有结果都是基于**三次重复实验**。 我们的评估机器有一个 Nvidia Tesla P100 GPU、112GB 的 RAM 和一个2.60GHz CPU (Intel E5-2690).。

对于NAO，它需要太多的计算资源，因此我们只使用提供 Pipeline 脚本的NAO-WS。

对于 Autkeras, 我们使用了它的 0.2.18 版本的代码, 因为它是我们开始实验时的最新版本。

## NAS 结果对比

| NAS             | AutoKeras (%) | ENAS (macro) (%) | ENAS (micro) (%) | DARTS (%) | NAO-WS (%) |
| --------------- |:-------------:|:----------------:|:----------------:|:---------:|:----------:|
| Fashion-MNIST   |     91.84     |      95.44       |      95.53       | **95.74** |   95.20    |
| CIFAR-10        |     75.78     |      95.68       |    **96.16**     |   94.23   |   95.64    |
| CIFAR-100       |     43.61     |      78.13       |      78.84       | **79.74** |   75.75    |
| OUI-Adience-Age |     63.20     |    **80.34**     |      78.55       |   76.83   |   72.96    |
| ImageNet-10-1   |     61.80     |      77.07       |      79.80       | **80.48** |   77.20    |
| ImageNet-10-2   |     37.20     |      58.13       |      56.47       |   60.53   | **61.20**  |

很遗憾，我们无法复现论文中所有的结果。

文章中提供的最佳或平均结果：

| NAS       | AutoKeras(%) | ENAS (macro) (%) | ENAS (micro) (%) |   DARTS (%)    | NAO-WS (%)  |
| --------- | ------------ |:----------------:|:----------------:|:--------------:|:-----------:|
| CIFAR- 10 | 88.56(best)  |   96.13(best)    |   97.11(best)    | 97.17(average) | 96.47(best) |

For AutoKeras, it has relatively worse performance across all datasets due to its random factor on network morphism.

For ENAS, ENAS (macro) shows good results in OUI-Adience-Age and ENAS (micro) shows good results in CIFAR-10.

For DARTS, it has a good performance on some datasets but we found its high variance in other datasets. The difference among three runs of benchmarks can be up to 5.37% in OUI-Adience-Age and 4.36% in ImageNet-10-1.

For NAO-WS, it shows good results in ImageNet-10-2 but it can perform very poorly in OUI-Adience-Age.

## 参考

1. Jin, Haifeng, Qingquan Song, and Xia Hu. "Efficient neural architecture search with network morphism." *arXiv preprint arXiv:1806.10282* (2018).

2. Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018).

3. Pham, Hieu, et al. "Efficient Neural Architecture Search via Parameters Sharing." international conference on machine learning (2018): 4092-4101.

4. Luo, Renqian, et al. "Neural Architecture Optimization." neural information processing systems (2018): 7827-7838.