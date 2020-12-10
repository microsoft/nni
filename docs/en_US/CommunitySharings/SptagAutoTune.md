# Automatically tuning SPTAG with NNI

[SPTAG](https://github.com/microsoft/SPTAG) (Space Partition Tree And Graph) is a library for large scale vector approximate nearest neighbor search scenario released by [Microsoft Research (MSR)](https://www.msra.cn/) and [Microsoft Bing](https://www.bing.com/).

This library assumes that the samples are represented as vectors and that the vectors can be compared by L2 distances or cosine distances. Vectors returned for a query vector are the vectors that have smallest L2 distance or cosine distances with the query vector.
SPTAG provides two methods: kd-tree and relative neighborhood graph (SPTAG-KDT) and balanced k-means tree and relative neighborhood graph (SPTAG-BKT). SPTAG-KDT is advantageous in index building cost, and SPTAG-BKT is advantageous in search accuracy in very high-dimensional data.

In SPTAG, there are tens of parameters that can be tuned for specified scenarios or datasets. NNI is a great tool for automatically tuning those parameters. The authors of SPTAG tried NNI for the auto tuning and found good-performing parameters easily, thus, they shared the practice of tuning SPTAG on NNI in their document [here](https://github.com/microsoft/SPTAG/blob/master/docs/Parameters.md). Please refer to it for detailed tutorial.