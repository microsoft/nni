Knowledge Distillation on NNI
=============================

KnowledgeDistill
----------------

Knowledge distillation support, in `Distilling the Knowledge in a Neural Network <https://arxiv.org/abs/1503.02531>`__\ ,  the compressed model is trained to mimic a pre-trained, larger model.  This training setting is also referred to as "teacher-student",  where the large model is the teacher and the small model is the student.


.. image:: ../../img/distill.png
   :target: ../../img/distill.png
   :alt: 


Usage
^^^^^

PyTorch code

.. code-block:: python

   from knowledge_distill.knowledge_distill import KnowledgeDistill
   kd = KnowledgeDistill(kd_teacher_model, kd_T=5)
   alpha = 1
   beta = 0.8
   for batch_idx, (data, target) in enumerate(train_loader):
       data, target = data.to(device), target.to(device)
       optimizer.zero_grad()
       output = model(data)
       loss = F.cross_entropy(output, target)
       # you only to add the following line to fine-tune with knowledge distillation
       loss = alpha * loss + beta * kd.loss(data=data, student_out=output)
       loss.backward()

User configuration for KnowledgeDistill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* **kd_teacher_model:** The pre-trained teacher model 
* **kd_T:** Temperature for smoothing teacher model's output

The complete code can be found `here <https://github.com/microsoft/nni/tree/v1.3/examples/model_compress/knowledge_distill/>`__
