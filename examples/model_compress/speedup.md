# Speedup Results

*This feature is still in Alpha version.*

This code only works on torch 1.3.1 and torchvision 0.4.2

## slim pruner example

on one V100 GPU,
input tensor: `torch.randn(64, 3, 32, 32)`

|Times| Mask Latency| Speedup Latency |
|---|---|---|
| 1 | 0.01197 | 0.005107 |
| 2 | 0.02019 | 0.008769 |
| 4 | 0.02733 | 0.014809 |
| 8 | 0.04310 | 0.027441 |
| 16 | 0.07731 | 0.05008 |
| 32 | 0.14464 | 0.10027 |

## fpgm pruner example

on cpu,
input tensor: `torch.randn(64, 1, 28, 28)`,
too large variance

|Times| Mask Latency| Speedup Latency |
|---|---|---|
| 1 | 0.01383 | 0.01839 |
| 2 | 0.01167 | 0.003558 |
| 4 | 0.01636 | 0.01088 |
| 40 | 0.14412 | 0.08268 |
| 40 | 1.29385 | 0.14408 |
| 40 | 0.41035 | 0.46162 |
| 400 | 6.29020 | 5.82143 |

## l1filter pruner example

on one V100 GPU,
input tensor: `torch.randn(64, 3, 32, 32)`

|Times| Mask Latency| Speedup Latency |
|---|---|---|
| 1 | 0.01026 | 0.003677 |
| 2 | 0.01657 | 0.008161 |
| 4 | 0.02458 | 0.020018 |
| 8 | 0.03498 | 0.025504 |
| 16 | 0.06757 | 0.047523 |
| 32 | 0.10487 | 0.086442 |

## APoZ pruner example

on one V100 GPU,
input tensor: `torch.randn(64, 3, 32, 32)`

|Times| Mask Latency| Speedup Latency |
|---|---|---|
| 1 | 0.01389 | 0.004208 |
| 2 | 0.01628 | 0.008310 |
| 4 | 0.02521 | 0.014008 |
| 8 | 0.03386 | 0.023923 |
| 16 | 0.06042 | 0.046183 |
| 32 | 0.12421 | 0.087113 |