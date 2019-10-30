# 运行模型压缩示例

以 PyTorch 剪枝为例：

```bash
python main_torch_pruner.py
```

此示例使用了 AGP Pruner。 初始化 Pruner 需要通过以下两种方式来提供配置。

- 读取 `configure_example.yaml`，这样代码会更整洁，但配置会比较复杂。
- 直接在代码中配置

此例在代码中配置了模型压缩：

```python
configure_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 0,
    'end_epoch': 10,
    'frequency': 1,
    'op_types': ['default']
}]
pruner = AGP_Pruner(configure_list)
```

当调用 `pruner(model)` 时，模型会被嵌入掩码操作。 例如，某层以权重作为输入，可在权重和层操作之间插入一个操作，此操作以权重为输入，并将其应用掩码后输出。 因此，计算过程中，只要通过此操作，就会应用掩码。 还可以**不做任何改动**，来对模型进行微调。

```python
for epoch in range(10):
    # update_epoch 来让 Pruner 知道 Epoch 的数量，从而能够在训练过程中调整掩码。
    pruner.update_epoch(epoch)
    print('# Epoch {} #'.format(epoch))
    train(model, device, train_loader, optimizer)
    test(model, device, test_loader)
```

微调完成后，被修剪过的权重可通过以下代码获得：

```
masks = pruner.mask_list
layer_name = xxx
mask = masks[layer_name]
```



