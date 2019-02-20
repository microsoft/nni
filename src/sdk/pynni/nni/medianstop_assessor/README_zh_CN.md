# Medianstop Assessor

## Median Stop

Medianstop 是一种简单的提前终止 Trial 的策略，可参考[论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)。 如果 Trial X 的在步骤 S 的最好目标值比所有已完成 Trial 的步骤 S 的中位数值明显要低，这个 Trial 就会被提前停止。