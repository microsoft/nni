# Grid Search

## Grid Search（遍历搜索）

Grid Search 会穷举定义在搜索空间文件中的所有超参组合。 注意，搜索空间仅支持 `choice`, `quniform`, `qloguniform`。 **The number `q` in `quniform` and `qloguniform` has special meaning (different from the spec in [search space spec](../../../../../docs/en_US/SearchSpaceSpec.md)). 这里的意义是在 `low` 和 `high` 之间均匀取值的数量。</p>