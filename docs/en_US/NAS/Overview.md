# NNI Programming Interface for Neural Architecture Search (NAS)

*This is an experimental feature, programming APIs are almost done, NAS trainers are under intensive development. ([NAS annotation](../AdvancedFeature/GeneralNasInterfaces.md) will become deprecated in future)*

Automatic neural architecture search is taking an increasingly important role on finding better models. Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually designed and tuned models. Some of representative works are [NASNet][2], [ENAS][1], [DARTS][3], [Network Morphism][4], and [Evolution][5]. There are new innovations keeping emerging. However, it takes great efforts to implement those algorithms, and it is hard to reuse code base of one algorithm for implementing another.

To facilitate NAS innovations (e.g., design/implement new NAS models, compare different NAS models side-by-side), an easy-to-use and flexible programming interface is crucial.

## Programming interface

A new programming interface for designing and searching for a model is often demanded in two scenarios.
    
    1. When designing a neural network, the designer may have multiple choices for a layer, sub-model, or connection, and not sure which one or a combination performs the best. It would be appealing to have an easy way to express the candidate layers/sub-models they want to try. 
    2. For the researchers who are working on automatic NAS, they want to have an unified way to express the search space of neural architectures. And making unchanged trial code adapted to different searching algorithms.

For expressing neural architecture search space, we provide two APIs:

```python
# choose one ``op`` from ``ops``, for pytorch this is a module.
# ops: for pytorch ``ops`` is a list of modules, for tensorflow it is a list of keras layers. An example in pytroch:
# ops = [PoolBN('max', channels, 3, stride, 1, affine=False),
#        PoolBN('avg', channels, 3, stride, 1, affine=False),
#        FactorizedReduce(channels, channels, affine=False),
#        SepConv(channels, channels, 3, stride, 1, affine=False),
#        DilConv(channels, channels, 3, stride, 2, 2, affine=False)]
# key: the name of this ``LayerChoice`` instance
nni.nas.LayerChoice(ops, key)
# choose ``n_selected`` from ``n_candidates`` inputs.
# n_candidates: the number of candidate inputs
# n_selected: the number of chosen inputs
# reduction: reduction operation for the chosen inputs
# key: the name of this ``InputChoice`` instance
nni.nas.InputChoice(n_candidates, n_selected, reduction, key)
```

After writing your model with search space embedded in the model using the above two APIs, the next step is finding the best model from the search space. Similar to optimizers of deep learning models, the procedure of finding the best model from search space can be viewed as a type of optimizing process, we call it `NAS trainer`. There have been several NAS trainers, for example, `DartsTrainer` which uses SGD to train architecture weights and model weights iteratively, `ENASTrainer` which uses a controller to train the model. New and more efficient NAS trainers keep emerging in research community.

NNI provides some popular NAS trainers, to use a NAS trainer, users could initialize a trainer after the model is defined:

```python
# create a DartsTrainer
trainer = DartsTrainer(model,
                       loss=criterion,
                       metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                       model_optim=optim,
                       lr_scheduler=lr_scheduler,
                       num_epochs=50,
                       dataset_train=dataset_train,
                       dataset_valid=dataset_valid,
                       batch_size=args.batch_size,
                       log_frequency=args.log_frequency)
# finding the best model from search space
trainer.train()
# export the best found model
trainer.export_model()
```

Different trainers could have different input arguments depending on their algorithms. After training, users could export the best one of the found models through `trainer.export_model()`.

[Here](https://github.com/microsoft/nni/blob/dev-nas-refactor/examples/nas/darts/main.py) is a trial example using DartsTrainer.

[1]: https://arxiv.org/abs/1802.03268
[2]: https://arxiv.org/abs/1707.07012
[3]: https://arxiv.org/abs/1806.09055
[4]: https://arxiv.org/abs/1806.10282
[5]: https://arxiv.org/abs/1703.01041