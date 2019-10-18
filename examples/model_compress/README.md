# Run model compression examples

You can run these examples easily like this, take torch pruning for example

```bash
python main_torch_pruner.py
```

Model compression can be configured in 2 ways

- By reading ```configure_example.yaml```, this can make codes clean when your configuration is complicated
- Directly config in your codes

In our example, we simply config model compression in our codes like this

```python
configure_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 1,
    'end_epoch': 11,
    'frequency': 1,
    'op_type': 'default'
}]
pruner = AGP_Pruner(configure_list)
```

Please notice that when ```pruner(model)``` called, our model compression codes will be **automatically injected** and you can fine-tune your model **without** any modifications, masked weights **won't** be updated any more during fine tuning.

```python
for epoch in range(10):
    print('# Epoch {} #'.format(epoch))
    train(model, device, train_loader, optimizer)
    test(model, device, test_loader)
    pruner.update_epoch(epoch + 1)
```

When fine tuning finished,  pruned weights are all masked and you can get masks like this

```
masks = pruner.mask_list
layer_name = xxx
mask = masks[layer_name]
```



