NNI 上的知识蒸馏
=============================

知识蒸馏 (Knowledge Distillation)
---------------------------------------

在 `Distilling the Knowledge in a Neural Network <https://arxiv.org/abs/1503.02531>`__\ 中提出了知识蒸馏（KD）的概念,  压缩后的模型被训练去模仿预训练的、较大的模型。  这种训练设置也称为"师生（teacher-student）"方式，其中大模型是教师，小模型是学生。 KD 通常用于微调剪枝后的模型。


.. image:: ../../img/distill.png
   :target: ../../img/distill.png
   :alt: 

用法
^^^^^

PyTorch 代码

.. code-block:: python

      for batch_idx, (data, target) in enumerate(train_loader):
         data, target = data.to(device), target.to(device)
         optimizer.zero_grad()
         y_s = model_s(data)
         y_t = model_t(data)
         loss_cri = F.cross_entropy(y_s, target)

         # kd 损失值
         p_s = F.log_softmax(y_s/kd_T, dim=1)
         p_t = F.softmax(y_t/kd_T, dim=1)
         loss_kd = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]

         # 总损失
         loss = loss_cir + loss_kd
         loss.backward()


微调剪枝模型的完整代码在 :githublink:`这里 <examples/model_compress/pruning/finetune_kd_torch.py>`

.. code-block:: python

      python finetune_kd_torch.py --model [model name] --teacher-model-dir [pretrained checkpoint path]  --student-model-dir [pruned checkpoint path] --mask-path [mask file path]

请注意：要微调剪枝后的模型，请先运行 :githublink:`basic_pruners_torch.py <examples/model_compress/pruning/basic_pruners_torch.py>` 来获取掩码文件，然后将掩码路径作为参数传递给脚本。


