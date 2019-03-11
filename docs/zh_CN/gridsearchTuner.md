# Grid Search

## Grid Search（遍历搜索）

Grid Search 会穷举定义在搜索空间文件中的所有超参组合。 注意，搜索空间仅支持 `choice`, `quniform`, `qloguniform`。 `quniform` 和 `qloguniform` 中的 **数字 `q` 有不同的含义（与[搜索空间](SearchSpaceSpec.md)说明不同）。 这里的意义是在 `low` 和 `high` 之间均匀取值的数量。</p>