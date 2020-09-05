# Run model compression examples

You can run these examples easily like this, take torch pruning for example

```bash
python main_torch_pruner.py
```

This example uses AGP Pruner. Initiating a pruner needs a user provided configuration which can be provided in two ways:

- By reading ```configure_example.yaml```, this can make code clean when your configuration is complicated
- Directly config in your codes

In our example, we simply config model compression in our codes like this

```python
configure_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 0,
    'end_epoch': 10,
    'frequency': 1,
    'op_types': ['default']
}]
pruner = AGPPruner(configure_list)
```

When ```pruner(model)``` is called, your model is injected with masks as embedded operations. For example, a layer takes a weight as input, we will insert an operation between the weight and the layer, this operation takes the weight as input and outputs a new weight applied by the mask. Thus, the masks are applied at any time the computation goes through the operations. You can fine-tune your model **without** any modifications.

```python
for epoch in range(10):
    # update_epoch is for pruner to be aware of epochs, so that it could adjust masks during training.
    pruner.update_epoch(epoch)
    print('# Epoch {} #'.format(epoch))
    train(model, device, train_loader, optimizer)
    test(model, device, test_loader)
```

When fine tuning finished,  pruned weights are all masked and you can get masks like this

```
masks = pruner.mask_list
layer_name = xxx
mask = masks[layer_name]
```



