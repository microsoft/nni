Medianstop Assessor
==========================

Median Stop
-----------

Medianstop 是一种简单的提前终止策略，可参考 `论文 <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf>`__。 如果 Trial X 在步骤 S 的最好目标值低于所有已完成 Trial 前 S 个步骤目标平均值的中位数，这个 Trial 就会被提前停止。
