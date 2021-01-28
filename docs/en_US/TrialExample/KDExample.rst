Knowledge Distillation on NNI
=============================

KnowledgeDistill
----------------

Knowledge Distillation (KD) is proposed in `Distilling the Knowledge in a Neural Network <https://arxiv.org/abs/1503.02531>`__\ ,  the compressed model is trained to mimic a pre-trained, larger model.  This training setting is also referred to as "teacher-student",  where the large model is the teacher and the small model is the student. KD is often used to fine-tune the pruned model.


.. image:: ../../img/distill.png
   :target: ../../img/distill.png
   :alt: 

Usage
^^^^^

PyTorch code

.. code-block:: python

      for batch_idx, (data, target) in enumerate(train_loader):
         data, target = data.to(device), target.to(device)
         optimizer.zero_grad()
         y_s = model_s(data)
         y_t = model_t(data)
         loss_cri = F.cross_entropy(y_s, target)

         # kd loss
         p_s = F.log_softmax(y_s/kd_T, dim=1)
         p_t = F.softmax(y_t/kd_T, dim=1)
         loss_kd = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]

         # total loss
         loss = loss_cir + loss_kd
         loss.backward()


The complete code for fine-tuning the pruend model can be found :githublink:`here <examples/model_compress/pruning/finetune_kd_torch.py>`

.. code-block:: python
      python finetune_kd_torch.py --model [model name] --teacher-model-dir [pretrained checkpoint path]  --student-model-dir [pruend checkpoint path] --mask-path [mask file path]


