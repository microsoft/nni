This README will be deleted once this hub got stabilized, after which we will promote it in the documentation.

## Why

We hereby provides a series of state-of-the-art search space, which is PyTorch model + mutations + training recipe.

For further motivations and plans, please see https://github.com/microsoft/nni/issues/4249.

## Reproduction Roadmap

1. Runnable
2. Load checkpoint of searched architecture and evaluate
3. Reproduce searched architecture
4. Runnable with built-in algos
5. Reproduce result with at least one algo

|                        | 1      | 2      | 3      | 4      | 5      |
|------------------------|--------|--------|--------|--------|--------|
| NasBench101            | Y      |        |        |        |        |
| NasBench201            | Y      |        |        |        |        |
| NASNet                 | Y      |        |        |        |        |
| ENAS                   | Y      |        |        |        |        |
| AmoebaNet              | Y      |        |        |        |        |
| PNAS                   | Y      |        |        |        |        |
| DARTS                  | Y      |        |        |        |        |
| ProxylessNAS           | Y      |        |        |        |        |
| MobileNetV3Space       | Y      |        |        |        |        |
| ShuffleNetSpace        | Y      |        |        |        |        |
| ShuffleNetSpace (ch)   | Y      |        |        |        |        |
