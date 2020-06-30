# NNI 上调优张量算子

## 概述

深度神经网络（DNN）的大量应用，催生了对从云服务器到嵌入式设备等不同硬件平台上的训练和推理的需求。 此外，还有深度神经网络上的计算图优化问题，如张量算子融合会引入新的张量算子。 然而，人工通过特定硬件库来优化张量算子，不能很好的支持新硬件平台和新算子，存在一定的局限性。因此，在大规模部署和深度学习技术的实际场景中，不同平台上的自动化的张量算子优化变得非常重要。

张量算子优化的本质，是一个组合优化问题。 The objective function is the performance of a tensor operator on specific hardware platform, which should be maximized with respect to the hyper-parameters of corresponding device code, such as how to tile a matrix or whether to unroll a loop. This example illustrates how to automatically tune tensor operators with NNI. Three tuning algorithms, OpEvo, G-BFS and N-A2C are provided. Please refer to [OpEvo: An Evolutionary Method for Tensor Operator Optimization](https://arxiv.org/abs/2006.05664) for detailed explanation about these algorithms.


## Environment Setup

We prepared a dockerfile for setting up experiment environments. Before starting, please make sure the Docker daemon is running and the driver of your GPU accelerator is properly installed. Enter into the example folder `examples/trials/systems/opevo` and run below command to build and instantiate a Docker image from the dockerfile.
```bash
# if you are using Nvidia GPU
make cuda-env
# if you are using AMD GPU
make rocm-env
```

## Run Experiments:

Three representative kinds of tensor operators, **matrix multiplication**, **batched matrix multiplication** and **2D convolution**, are chosen from BERT and AlexNet, and tuned with NNI. The `Trial` code for all tensor operators is `/root/compiler_auto_tune_stable.py`, and `Search Space` files and `config` files for each tuning algorithm locate in `/root/experiments/`, which are categorized by tensor operators. Here `/root` refers to the root of the container.

For tuning the operators of matrix multiplication, please run below commands from `/root`:
```bash
# (N, K) x (K, M) 表示形状为 (N, K) 的矩阵乘以形状为 (K, M) 的矩阵

# (512, 1024) x (1024, 1024)
# 用 opevo 调优
nnictl create --config experiments/mm/N512K1024M1024/config_opevo.yml
# 用 g-bfs 调优
nnictl create --config experiments/mm/N512K1024M1024/config_gbfs.yml
# 用 n-a2c 调优
nnictl create --config experiments/mm/N512K1024M1024/config_na2c.yml

# (512, 1024) x (1024, 4096)
# 用 opevo 调优
nnictl create --config experiments/mm/N512K1024M4096/config_opevo.yml
# 用 g-bfs 调优
nnictl create --config experiments/mm/N512K1024M4096/config_gbfs.yml
# 用 n-a2c 调优
nnictl create --config experiments/mm/N512K1024M4096/config_na2c.yml

# (512, 4096) x (4096, 1024)
# 用 opevo 调优
nnictl create --config experiments/mm/N512K1024M4096/config_opevo.yml
# 用 g-bfs 调优
nnictl create --config experiments/mm/N512K1024M4096/config_gbfs.yml
# 用 n-a2c 调优
nnictl create --config experiments/mm/N512K1024M4096/config_na2c.yml
```

For tuning the operators of batched matrix multiplication, please run below commands from `/root`:
```bash
# batched matrix with batch size 960 and shape of matrix (128, 128) multiplies batched matrix with batch size 960 and shape of matrix (128, 64)
nnictl create --config experiments/bmm/B960N128K128M64PNN/config_opevo.yml
# batched matrix with batch size 960 and shape of matrix (128, 128) is transposed first and then multiplies batched matrix with batch size 960 and shape of matrix (128, 64)
nnictl create --config experiments/bmm/B960N128K128M64PTN/config_opevo.yml
# batched matrix with batch size 960 and shape of matrix (128, 64) is transposed first and then right multiplies batched matrix with batch size 960 and shape of matrix (128, 64).
nnictl create --config experiments/bmm/B960N128K64M128PNT/config_opevo.yml
```

For tuning the operators of 2D convolution, please run below commands from `/root`:
```bash
# image tensor of shape $(512, 3, 227, 227)$ convolves with kernel tensor of shape $(64, 3, 11, 11)$ with stride 4 and padding 0
nnictl create --config experiments/conv/N512C3HW227F64K11ST4PD0/config_opevo.yml
# image tensor of shape $(512, 64, 27, 27)$ convolves with kernel tensor of shape $(192, 64, 5, 5)$ with stride 1 and padding 2
nnictl create --config experiments/conv/N512C64HW27F192K5ST1PD2/config_opevo.yml
```

Please note that G-BFS and N-A2C are not eligible to tune the operators of batched matrix multiplication and 2D convolution, since there are unsupportable parameters in the search spaces of these operators.

## Citing OpEvo

If you use OpEvo in your research, please consider citing the paper as follows:
```
@misc{gao2020opevo,
    title={OpEvo: An Evolutionary Method for Tensor Operator Optimization},
    author={Xiaotian Gao and Cui Wei and Lintao Zhang and Mao Yang},
    year={2020},
    eprint={2006.05664},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
