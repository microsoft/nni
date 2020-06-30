# NNI 上调优张量算子

## 概述

深度神经网络（DNN）的大量应用，催生了对从云服务器到嵌入式设备等不同硬件平台上的训练和推理的需求。 此外，还有深度神经网络上的计算图优化问题，如张量算子融合会引入新的张量算子。 然而，人工通过特定硬件库来优化张量算子，不能很好的支持新硬件平台和新算子，存在一定的局限性。因此，在大规模部署和深度学习技术的实际场景中，不同平台上的自动化的张量算子优化变得非常重要。

张量算子优化的本质，是一个组合优化问题。 目标函数是张量算子在某个硬件平台上的性能，通过调整超参（如，如何切片矩阵，展开循环）实现该平台上的最佳性能。 此示例展示了如何使用 NNI 来自动调优张量算子。 其提供了 OpEvo, G-BFS 和 N-A2C 三种调优算法。 参考论文 [OpEvo: An Evolutionary Method for Tensor Operator Optimization](https://arxiv.org/abs/2006.05664) 来了解算法。


## 配置环境

此示例准备了 Dockerfile 作为 Experiment 的环境。 开始前，确保 Docker 守护进程已启动，GPU 加速驱动已正确安装。 进入示例文件夹 `examples/trials/systems/opevo` 并运行下列命令，从 Dockerfile 构建并实例化 Docker 映像。
```bash
# 如果使用 Nvidia GPU
make cuda-env
# 如果使用 AMD GPU
make rocm-env
```

## 运行 Experiment

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
