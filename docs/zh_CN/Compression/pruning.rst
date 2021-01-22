#################
剪枝
#################

剪枝是一种常用的神经网络模型压缩技术。
剪枝算法探索模型权重（参数）中的冗余，并尝试去除冗余和非关键权重，
将它们的值归零，确保其不参与反向传播过程。

从剪枝粒度的角度来看，细粒度剪枝或非结构化剪枝是指分别对每个权重进行剪枝。
粗粒度剪枝或结构化剪枝是修剪整组权重，例如卷积滤波器。

NNI 提供了多种非结构化和结构化剪枝算法。
其使用了统一的接口来支持 TensorFlow 和 PyTorch。
只需要添加几行代码即可压缩模型。
对于结构化滤波器剪枝，NNI 还提供了依赖感知模式。 在依赖感知模式下，
滤波器剪枝在加速后会获得更好的速度增益。

详细信息，参考以下教程：

..  toctree::
    :maxdepth: 2

    Pruners <Pruner>
    Dependency Aware Mode <DependencyAware>
    Model Speedup <ModelSpeedup>
    Automatic Model Pruning with NNI Tuners <AutoPruningUsingTuners>
