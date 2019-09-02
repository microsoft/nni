# 使用 NNI 为 SPTAG 自动调参

[SPTAG](https://github.com/microsoft/SPTAG) (Space Partition Tree And Graph) 是大规模向量的最近邻搜索的工具，由[微软研究院（MSR）](https://www.msra.cn/)和[微软必应团队](https://www.bing.com/)联合发布。

此工具假设样本可以表示为向量，并且能通过 L2 或余弦算法来比较距离。 输入一个查询向量，会返回与其 L2 或余弦距离最小的一组向量。 SPTAG 提供了两种方法：kd-tree 与其的相关近邻图 (SPTAG-KDT)，以及平衡 k-means 树与其的相关近邻图 （SPTAG-BKT）。 SPTAG-KDT 在索引构建效率上较好，而 SPTAG-BKT 在搜索高维度数据的精度上较好。

在 SPTAG中，有几十个参数可以根据特定的场景或数据集进行调优。 NNI 是用来自动化调优这些参数的绝佳工具。 SPTAG 的作者尝试了使用 NNI 来进行自动调优，并轻松找到了性能较好的参数组合，并在 SPTAG [文档](https://github.com/microsoft/SPTAG/blob/master/docs/Parameters.md)中进行了分享。 参考此文档了解详细教程。