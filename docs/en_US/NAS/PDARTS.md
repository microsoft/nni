# P-DARTS

## Examples

[Example code](https://github.com/microsoft/nni/tree/v1.9/examples/nas/pdarts)

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/pdarts
python3 search.py

# train the best architecture, it's the same progress as darts.
cd ../darts
python3 retrain.py --arc-checkpoint ../pdarts/checkpoints/epoch_2.json
```
