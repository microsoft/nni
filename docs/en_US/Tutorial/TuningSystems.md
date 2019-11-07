# Automatically tune systems with NNI

As computer systems and networking get increasingly complicated, optimizing them manually with explicit rules and heuristics becomes harder than ever before, sometimes impossible. At Microsoft Research Asia, our AutoSys project applies learning techniques provided by NNI to large-scale system performance tuning. Based on AutoSys, we have built a tool to help many crucial system scenarios within Microsoft. These scenarios include multimedia search for Bing (e.g., tail latency reduced by up to ~40%, and capacity increased by up to ~30%), job scheduling for Bing Ads (e.g., tail latency reduced by up to ~13%), and so on.

Below are two examples of tuning systems with NNI. Anyone can easily tune their own systems by following them.

* [Tuning RocksDB with NNI](../TrialExample/RocksdbExamples.md)
* [Tuning parameters of SPTAG (Space Partition Tree And Graph) with NNI](https://github.com/microsoft/SPTAG/blob/master/docs/Parameters.md)