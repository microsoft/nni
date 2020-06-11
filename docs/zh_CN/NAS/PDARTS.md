# P-DARTS

## 示例

[示例代码](https://github.com/microsoft/nni/tree/master/examples/nas/pdarts)

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# 搜索最好的架构
cd examples/nas/pdarts
python3 search.py

# 训练最好的架构，过程与 darts 相同。
cd ../darts
python3 retrain.py --arc-checkpoint ../pdarts/checkpoints/epoch_2.json
```
