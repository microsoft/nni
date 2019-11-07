# 使用 NNI 自动为系统调参

随着计算机系统和网络变得日益复杂，通过显式的规则和启发式的方法对它们进行手工优化变得越来越难，甚至是不可能的。在微软亚洲研究院，我们的 AutoSys 项目通过应用 NNI 提供的机器学习技术根据系统性能来自动调整系统的运行参数。AutoSys 项目已经成功应用在了微软内部很多重要的系统场景当中。这些场景包括必应的多媒体搜索 （尾延迟减少约 40%，容量提高约30%），必应广告的任务调度（尾延迟减少约 13%），等。

下面是两个使用 NNI 自动调整系统运行参数的例子。通过参考这些例子，人们可以方便地使用 NNI 来为自己的系统自动调优。

* [使用 NNI 调优 RocksDB](../TrialExample/RocksdbExamples.md)
* [使用 NNI 调优 SPTAG (Space Partition Tree And Graph)](https://github.com/microsoft/SPTAG/blob/master/docs/Parameters.md)