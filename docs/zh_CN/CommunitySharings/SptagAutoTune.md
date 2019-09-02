# 使用 NNI 为 SPTAG 自动调参

[SPTAG](https://github.com/microsoft/SPTAG) (Space Partition Tree And Graph) 是大规模向量的最近邻搜索的工具，由[微软研究院（MSR）](https://www.msra.cn/)和[微软必应团队](https://www.bing.com/)联合发布。

此工具假设样本可以表示为向量，并且能通过 L2 或余弦算法来比较距离。 Vectors returned for a query vector are the vectors that have smallest L2 distance or cosine distances with the query vector. SPTAG provides two methods: kd-tree and relative neighborhood graph (SPTAG-KDT) and balanced k-means tree and relative neighborhood graph (SPTAG-BKT). SPTAG-KDT is advantageous in index building cost, and SPTAG-BKT is advantageous in search accuracy in very high-dimensional data.

In SPTAG, there are tens of parameters that can be tuned for specified scenarios or datasets. NNI is a great tool for automatically tuning those parameters. The authors of SPTAG tried NNI for the auto tuning and found good-performing parameters easily, thus, they shared the practice of tuning SPTAG on NNI in their document [here](https://github.com/microsoft/SPTAG/blob/master/docs/Parameters.md). Please refer to it for detailed tutorial.