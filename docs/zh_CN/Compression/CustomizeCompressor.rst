自定义压缩算法
===================================

.. contents::

为了简化实现新压缩算法的过程，NNI 设计了简单灵活，同时支持剪枝和量化的接口。 首先会介绍如何自定义新的剪枝算法，然后介绍如何自定义新的量化算法。

**重要说明**，为了更好的理解如何定制新的剪枝、量化算法，应先了解 NNI 中支持各种剪枝算法的框架。 参考 `模型压缩框架概述 </Compression/Framework.html>`__。

自定义剪枝算法
---------------------------------

要实现新的剪枝算法，需要实现 ``权重掩码`` 类，它是 ``WeightMasker`` 的子类，以及 ``Pruner`` 类，它是 ``Pruner`` 的子类。

``权重掩码`` 的实现如下：

.. code-block:: python

   class MyMasker(WeightMasker):
       def __init__(self, model, pruner):
           super().__init__(model, pruner)
           # 此处可初始化，如为算法收集计算权重所需要的统计信息。
           # 如果你的算法需要计算掩码

       def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
           # 根据 wrapper.weight, 和 sparsity,  
           # 及其它信息来计算掩码
           # mask = ...
           return {'weight_mask': mask}

参考 NNI 提供的 :githublink:`权重掩码 <src/sdk/pynni/nni/compression/pytorch/pruning/structured_pruning.py>` 来实现自己的权重掩码。

基础的 ``Pruner`` 如下所示：

.. code-block:: python

   class MyPruner(Pruner):
       def __init__(self, model, config_list, optimizer):
           super().__init__(model, config_list, optimizer)
           self.set_wrappers_attribute("if_calculated", False)
           # 创建权重掩码实例
           self.masker = MyMasker(model, self)

       def calc_mask(self, wrapper, wrapper_idx=None):
           sparsity = wrapper.config['sparsity']
           if wrapper.if_calculated:
               # 如果是一次性剪枝算法，不需要再次剪枝
               return None
           else:
               # 调用掩码函数来实际计算当前层的掩码
               masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)
               wrapper.if_calculated = True
               return masks

参考 NNI 提供的 :githublink:`Pruner <src/sdk/pynni/nni/compression/pytorch/pruning/one_shot.py>` 来实现自己的 Pruner。

----

自定义量化算法
--------------------------------------

要实现新的量化算法，需要继承 ``nni.compression.pytorch.Quantizer``。 然后，根据算法逻辑来重写成员函数。 需要重载的成员函数是 ``quantize_weight``。 ``quantize_weight`` 直接返回量化后的权重，而不是 mask。这是因为对于量化算法，量化后的权重不能通过应用 mask 来获得。

.. code-block:: python

   from nni.compression.pytorch import Quantizer

   class YourQuantizer(Quantizer):
       def __init__(self, model, config_list):
           """
           建议使用 NNI 定义的规范来配置
           """
           super().__init__(model, config_list)

       def quantize_weight(self, weight, config, **kwargs):
           """
           quantize 需要重载此方法来为权重提供掩码
           此方法挂载于模型的 :meth:`forward`。

           参数
           ----------
           weight : Tensor
               要被量化的权重
           config : dict
               输出量化的配置
           """

           # 此处逻辑生成 `new_weight`

           return new_weight

       def quantize_output(self, output, config, **kwargs):
           """
           重载此方法量化输入
           此方法挂载于模型的 `:meth:`forward`。

           参数量
           ----------
           output : Tensor
               需要被量化的输出
           config : dict
               输出量化的配置
           """

           # 生成 `new_output` 的代码

           return new_output

       def quantize_input(self, *inputs, config, **kwargs):
           """
           重载此方法量化输入
           此方法挂载于模型的 :meth:`forward`。

           参数量
           ----------
           inputs : Tensor
               需要被量化的张量
           config : dict
               输入量化的配置
           """

           # 生成 `new_input` 的代码

           return new_input

       def update_epoch(self, epoch_num):
           pass

       def step(self):
           """
           根据 bind_model 函数传入的模型或权重
           进行一些处理
           """
           pass

定制 backward 函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^

有时，量化操作必须自定义 backward 函数，例如 `Straight-Through Estimator <https://stackoverflow.com/questions/38361314/the-concept-of-straight-through-estimator-ste>`__\ ，可如下定制 backward 函数：

.. code-block:: python

   from nni.compression.pytorch.compressor import Quantizer, QuantGrad, QuantType

   class ClipGrad(QuantGrad):
       @staticmethod
       def quant_backward(tensor, grad_output, quant_type):
           """
           此方法应被子类重载来提供定制的 backward 函数，
           默认实现是 Straight-Through Estimator
           Parameters
           ----------
           tensor : Tensor
               量化操作的输入
           grad_output : Tensor
               量化操作输出的梯度
           quant_type : QuantType
               量化类型，可被定义为 `QuantType.QUANT_INPUT`, `QuantType.QUANT_WEIGHT`, `QuantType.QUANT_OUTPUT`,
               可为不同的类型定义不同的行为。
           Returns
           -------
           tensor
               量化输入的梯度
           """

           # 对于 quant_output 函数，如果张量的绝对值大于 1，则将梯度设置为 0
           if quant_type == QuantType.QUANT_OUTPUT: 
               grad_output[torch.abs(tensor) > 1] = 0
           return grad_output


   class YourQuantizer(Quantizer):
       def __init__(self, model, config_list):
           super().__init__(model, config_list)
           # 定制 backward 函数来重载默认的 backward 函数
           self.quant_grad = ClipGrad

如果不定制 ``QuantGrad``，默认的 backward 为 Straight-Through Estimator。 
*编写中*……
