P-DARTS
=======

示例
--------

:githublink:`示例代码 <examples/nas/pdarts>`

.. code-block:: bash

   ＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
   git clone https://github.com/Microsoft/nni.git

   # 搜索最优结构
   cd examples/nas/pdarts
   python3 search.py

   # 训练最优架构，跟 darts 一样的步骤
   cd ../darts
   python3 retrain.py --arc-checkpoint ../pdarts/checkpoints/epoch_2.json
