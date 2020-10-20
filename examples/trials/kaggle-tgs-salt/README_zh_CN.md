## Kaggle 比赛 [TGS Salt Identification Chanllenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) 第 33 名的解决方案

本示例展示了如何不改动代码的情况下通过 NNI 来为竞赛代码使用自动机器学习。 要在 NNI 上运行此代码，首先需要单独运行它，然后配置 config.yml：

    nnictl create --config config.yml


此代码仍然能够单独运行，但需要至少一周来重现竞赛的结果。

[解决方案概述](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69593)

准备：

下载完整的数据，运行 preprocess.py 来准备数据。

阶段 1：

将目录 0-3 训练 100 个 epoch，对于每个目录，训练三个模型：

    python3 train.py --ifolds 0 --epochs 100 --model_name UNetResNetV4
    python3 train.py --ifolds 0 --epochs 100 --model_name UNetResNetV5 --layers 50
    python3 train.py --ifolds 0 --epochs 100 --model_name UNetResNetV6


阶段 2：

使用余弦退火学习率调度器运行 300 次 epoch 来微调阶段 1 的模型：

    python3 train.py --ifolds 0 --epochs 300 --lrs cosine --lr 0.001 --min_lr 0.0001 --model_name UNetResNetV4


阶段 3：

用深度通道微调阶段 2 的模型：

    python3 train.py --ifolds 0 --epochs 300 --lrs cosine --lr 0.001 --min_lr 0.0001 --model_name UNetResNetV4 --depths


阶段 4：

为每个模型进行预测，组合结果生成伪标签。

阶段 5：

用伪标签微调阶段 3 的模型

    python3 train.py --ifolds 0 --epochs 300 --lrs cosine --lr 0.001 --min_lr 0.0001 --model_name UNetResNetV4 --depths --pseudo


阶段 6： 将所有阶段 3 和阶段 5 的模型组合起来。