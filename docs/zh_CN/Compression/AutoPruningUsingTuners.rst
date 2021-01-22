使用 NNI Tuners 自动进行模型压缩
========================================

使用 NNI 能轻松实现自动模型压缩

首先，使用 NNI 压缩模型
---------------------------------

可使用 NNI 轻松压缩模型。 以剪枝为例，可通过 LevelPruner 对预训练模型剪枝：

.. code-block:: python

   from nni.algorithms.compression.pytorch.pruning import LevelPruner
   config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
   pruner = LevelPruner(model, config_list)
   pruner.compress()

op_type 为 'default' 表示模块类型为 PyTorch 定义在了 :githublink:`default_layers.py <src/sdk/pynni/nni/compression/pytorch/default_layers.py>` 。

因此 ``{ 'sparsity': 0.8, 'op_types': ['default'] }`` 表示 **所有指定 op_types 的层都会被压缩到 0.8 的稀疏度**。 当调用 ``pruner.compress()`` 时，模型会通过掩码进行压缩。随后还可以微调模型，此时 **被剪除的权重不会被更新**。

然后，进行自动化
-------------------------

前面的示例人工选择了 LevelPruner，并对所有层使用了相同的稀疏度，显然这不是最佳方法，因为不同层会有不同的冗余度。 每层的稀疏度都应该仔细调整，以便减少模型性能的下降，可通过 NNI Tuner 来完成。

首先需要设计搜索空间，这里使用了嵌套的搜索空间，其中包含了选择的剪枝函数以及需要优化稀疏度的层。

.. code-block:: json

   {
     "prune_method": {
       "_type": "choice",
       "_value": [
         {
           "_name": "agp",
           "conv0_sparsity": {
             "_type": "uniform",
             "_value": [
               0.1,
               0.9
             ]
           },
           "conv1_sparsity": {
             "_type": "uniform",
             "_value": [
               0.1,
               0.9
             ]
           },
         },
         {
           "_name": "level",
           "conv0_sparsity": {
             "_type": "uniform",
             "_value": [
               0.1,
               0.9
             ]
           },
           "conv1_sparsity": {
             "_type": "uniform",
             "_value": [
               0.01,
               0.9
             ]
           },
         }
       ]
     }
   }

然后需要修改几行代码。

.. code-block:: python

   import nni
   from nni.algorithms.compression.pytorch.pruning import *
   params = nni.get_parameters()
   conv0_sparsity = params['prune_method']['conv0_sparsity']
   conv1_sparsity = params['prune_method']['conv1_sparsity']
   # 如果需要约束总稀疏度，则应缩放原始稀疏度
   config_list_level = [{ 'sparsity': conv0_sparsity, 'op_name': 'conv0' },
                        { 'sparsity': conv1_sparsity, 'op_name': 'conv1' }]
   config_list_agp = [{'initial_sparsity': 0, 'final_sparsity': conv0_sparsity,
                       'start_epoch': 0, 'end_epoch': 3,
                       'frequency': 1,'op_name': 'conv0' },
                      {'initial_sparsity': 0, 'final_sparsity': conv1_sparsity,
                       'start_epoch': 0, 'end_epoch': 3,
                       'frequency': 1,'op_name': 'conv1' },]
   PRUNERS = {'level':LevelPruner(model, config_list_level), 'agp':AGPPruner(model, config_list_agp)}
   pruner = PRUNERS(params['prune_method']['_name'])
   pruner.compress()
   ... # 微调
   acc = evaluate(model) # evaluation
   nni.report_final_results(acc)

最后，定义任务，并使用任务来自动修剪层稀疏度。

.. code-block:: yaml

   authorName: default
   experimentName: Auto_Compression
   trialConcurrency: 2
   maxExecDuration: 100h
   maxTrialNum: 500
   #choice: local, remote, pai
   trainingServicePlatform: local
   #choice: true, false
   useAnnotation: False
   searchSpacePath: search_space.json
   tuner:
     #choice: TPE, Random, Anneal...
     builtinTunerName: TPE
     classArgs:
       #choice: maximize, minimize
       optimize_mode: maximize
   trial:
     command: bash run_prune.sh
     codeDir: .
     gpuNum: 1
