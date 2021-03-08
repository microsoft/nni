Neural Architecture Search with Retiarii (Experimental)
=======================================================

`Retiarii <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ is a new framework to support neural architecture search and hyper-parameter tuning. It allows users to express various search space with high flexibility, to reuse many SOTA search algorithms, and to leverage system level optimizations to speed up the search process. This framework provides the following new user experiences.

* Search space can be expressed directly in user model code. A tuning space can be expressed along defining a model.
* Neural architecture candidates and hyper-parameter candidates are more friendly supported in an experiment.
* The experiment can be launched directly from python code.

*We are working on migrating* `our previous NAS framework <../Overview.rst>`__ *to Retiarii framework. Thus, this feature is still experimental. We recommend users to try the new framework and provide your valuable feedback for us to improve it. The old framework is still supported for now.*

.. contents::

There are mainly two crucial components for a neural architecture search task, namely,

* Model search space that defines the set of models to explore.
* A proper strategy as the method to explore this search space.
* A model evaluator that reports the performance of a given model.

.. note:: Currently, PyTorch is the only supported framework by Retiarii, and we have only tested with PyTorch 1.6 and 1.7. This documentation assumes PyTorch context but it should also apply to other frameworks, that is in our future plan.

Define your Model Space
-----------------------

Model space is defined by users to express a set of models that users want to explore, which should contains potentially good-performing models. In this framework, a model space is defined with two parts: a base model and possible mutations on the base model.

Define Base Model
^^^^^^^^^^^^^^^^^

Defining a base model is almost the same as defining a PyTorch (or TensorFlow) model.

However, base model needs to be, at least, slightly modified to make mutations work. There are basically two constraints:

* The underlying implementation of Retiarii converts base model into a graph, runs mutators on the converted graph, and converts the mutated graph back to a model that is supported by deep learning frameworks. Therefore, the base model needs to be supported by the converter. For example, in PyTorch, TorchScript needs to be able to recognize and parse this model.
* Deep learning models tend to be a hierachical structure and tends to be very complicated if expanded. For example, a transformer contains several encoders and decoders. An encoder contains a self-attention. An attention layer contains several linear layer. If the only intention is to mutate a parameter on the transformer, there is no need to expand the full graph. To this end, Retiarii considers mutated modules as the most basic building blocks and does not expand them any more. So far, users need to manually annotate them with ``@basic_unit`` For example, user-defined modules used in ``LayerChoice`` should be decorated. Users can refer to `here <#serializer>`__ on the detailed usages.

Below is a very simple example of defining a base model, it is almost the same as defining a PyTorch model.

.. code-block:: python

  import torch.nn.functional as F
  import nni.retiarii.nn.pytorch as nn

  @basic_unit
  class BasicBlock(nn.Module):
    def __init__(self, const):
      self.const = const
    def forward(self, x):
      return x + self.const

  class ConvPool(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 1, 5)  # possibly mutate this conv
      self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
      return self.pool(self.conv(x))

  class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.convpool = ConvPool()
      self.mymodule = BasicBlock(2.)
    def forward(self, x):
      return F.relu(self.convpool(self.mymodule(x)))

Users can refer to :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>` and :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>` for more complicated examples.

Define Model Mutations
^^^^^^^^^^^^^^^^^^^^^^

A base model is only one concrete model not a model space. We provide APIs and primitives for users to express how the base model can be mutated, i.e., a model space which includes many models. We introduce two ways to write mutations. These two methods are mutually exclusive, meaning that you cannot use both of them in one model.

**Inline mutations: express mutations in an inlined manner**

For easy usability and also backward compatibility, we provide some APIs for users to easily express possible mutations after defining a base model. The APIs can be used just like PyTorch module.

* ``nn.LayerChoice``. It allows users to put several candidate operations (e.g., PyTorch modules), one of them is chosen in each explored model. *Note that if the candidate is a user-defined module, it should be decorated as `serialize module <#serializer>`__. In the following example, ``ops.PoolBN`` and ``ops.SepConv`` should be decorated.*

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__`
    self.layer = nn.LayerChoice([
      ops.PoolBN('max', channels, 3, stride, 1),
      ops.SepConv(channels, channels, 3, stride, 1),
      nn.Identity()
    ]))
    # invoked in `forward` function
    out = self.layer(x)

* ``nn.InputChoice``. It is mainly for choosing (or trying) different connections. It takes several tensors and chooses ``n_chosen`` tensors from them.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # declared in `__init__`
    self.input_switch = nn.InputChoice(n_chosen=1)
    # invoked in `forward` function, choose one from the three
    out = self.input_switch([tensor1, tensor2, tensor3])

* ``nn.ValueChoice``. It is for choosing one value from some candidate values. It can only be used as input argument of basic units, that is, modules in ``nni.retiarii.nn.pytorch`` and user-defined modules decorated with ``@basic_unit``.

  .. code-block:: python

    # import nni.retiarii.nn.pytorch as nn
    # used in `__init__`
    self.conv = nn.Conv2d(XX, XX, kernel_size=nn.ValueChoice([1, 3, 5])
    self.op = MyOp(nn.ValueChoice([0, 1]), nn.ValueChoice([-1, 1]))

All the APIs have an optional argument called ``label`` and mutations with the same label will share the same choice. A typical example is,

  .. code-block:: python

    self.net = nn.Sequential(
        nn.Linear(10, nn.ValueChoice([32, 64, 128], label='hidden_dim'),
        nn.Linear(nn.ValueChoice([32, 64, 128], label='hidden_dim'), 3)
    )

Detailed API description and usage can be found `here <./ApiReference.rst>`__\. Example of using these APIs can be found in :githublink:`Darts base model <test/retiarii_test/darts/darts_model.py>`. We are actively enrich the set of inline mutations, to make it easier to express a new search space.

**Express mutations with mutators**

Though easy-to-use, inline mutations have limited expressiveness, some model spaces cannot be expressed. To improve expressiveness and flexibility, we provide primitives for users to write *Mutator* to express how they want to mutate base model more flexibly. Mutator stands above base model, thus has full ability to edit the model.

Users can instantiate several mutators as below, the mutators will be sequentially applied to the base model one after another for sampling a new model.

.. code-block:: python

  applied_mutators = []
  applied_mutators.append(BlockMutator('mutable_0'))
  applied_mutators.append(BlockMutator('mutable_1'))

``BlockMutator`` is defined by users to express how to mutate the base model. User-defined mutator should inherit ``Mutator`` class, and implement mutation logic in the member function ``mutate``.

.. code-block:: python

  from nni.retiarii import Mutator
  class BlockMutator(Mutator):
    def __init__(self, target: str, candidates: List):
        super(BlockMutator, self).__init__()
        self.target = target
        self.candidate_op_list = candidates

    def mutate(self, model):
      nodes = model.get_nodes_by_label(self.target)
      for node in nodes:
        chosen_op = self.choice(self.candidate_op_list)
        node.update_operation(chosen_op.type, chosen_op.params)

The input of ``mutate`` is graph IR of the base model (please refer to `here <./ApiReference.rst>`__ for the format and APIs of the IR), users can mutate the graph with its member functions (e.g., ``get_nodes_by_label``, ``update_operation``). The mutation operations can be combined with the API ``self.choice``, in order to express a set of possible mutations. In the above example, the node's operation can be changed to any operation from ``candidate_op_list``.

Use placehoder to make mutation easier: ``nn.Placeholder``. If you want to mutate a subgraph or node of your model, you can define a placeholder in this model to represent the subgraph or node. Then, use mutator to mutate this placeholder to make it real modules.

.. code-block:: python

  ph = nn.Placeholder(
    label='mutable_0',
    kernel_size_options=[1, 3, 5],
    n_layer_options=[1, 2, 3, 4],
    exp_ratio=exp_ratio,
    stride=stride
  )

``label`` is used by mutator to identify this placeholder. The other parameters are the information that are required by mutator. They can be accessed from ``node.operation.parameters`` as a dict, it could include any information that users want to put to pass it to user defined mutator. The complete example code can be found in :githublink:`Mnasnet base model <test/retiarii_test/mnasnet/base_mnasnet.py>`.

Explore the Defined Model Space
-------------------------------

After model space is defined, it is time to explore this model space. Users can choose a proper search strategy to explore the model space.

Retiarii currently supports the following search strategies:

* Grid search: enumerate all the possible models defined in the space.
* Random: randomly pick the models from search space.
* Regularized evolution: a genetic algorithm that explores the space based on inheritance and mutation.

The built-in search strategies should work well on spaces that is defined with inline mutations, though users need to be aware that not every search strategy can be applied to every model space.

Create a search strategy is very easy. An example is as follows,

.. code-block:: python

  import nni.retiarii.strategy as strategy

  search_strategy = strategy.Random(dedup=True)  # dedup=False if deduplication is not wanted

Detailed descriptions and usages can be found `here <./ApiReference.rst>`__ .

Evaluate a Specific Model
-------------------------

In the NAS process, the search strategy repeatedly generates new models, and another component called model performance evaluator repeatedly trains and validates those generated models. The obtained performances are collected and sent to search strategy as feedbacks to help the strategy make better decisions.

The model evaluator should correctly identify the use scenario of the model and the optimization goal. For example, on a classification task, a input-label dataset is needed, the loss function might be cross entropy and the optimized metric could be accuracy. On a regression task, the optimized metric could be mean-squared-error. Other situations could be more complicated, for example, object detection, GAN or RL. In a word, a model evaluator should define everything other than the model itself. It should be as self-contained as possible.

In the context of PyTorch, Retiarii has provided two built-in model evaluators, designed for simple use cases: classification and regression. These two evaluators and built upon the awesome library PyTorch-Lightning. An example here implements a simple evaluator that runs on MNIST dataset, trains for 10 epochs, and reports its validation accuracy.

.. code-block:: python

  import nni.retiarii.evaluator.pytorch.lightning as pl
  from nni.retiarii import serialize
  from torchvision import transforms

  transform = serialize(transforms.Compose, [serialize(transforms.ToTensor()), serialize(transforms.Normalize, (0.1307,), (0.3081,))])
  train_dataset = serialize(MNIST, root='data/mnist', train=True, download=True, transform=transform)
  test_dataset = serialize(MNIST, root='data/mnist', train=False, download=True, transform=transform)
  evaluator = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

As the model evaluator is running in another process (possibly in some remote machines), the defined evaluators, along with all its parameters, needs to be correctly serialized. For example, users should use the dataloader that has been already wrapped as a serializable class defined in ``nni.retiarii.evaluator.pytorch.lightning``. For the arguments used in dataloader, recursive serialization needs to be done, until it belongs to one of the primitive types like int, str, float.

In case the use scenario is more complicated than the provided evaluators, or the training code is already available and all you want is to run a function, please refer to `the guide to write a new evaluator <./WriteEvaluator.rst>`__ .

.. note:: In case you want to run the model evaluator locally for debugging purposes, you can directly run the evaluator via ``evaluator._execute(Net)`` (note that it has to be ``Net``, not ``Net()``). However, this API is currently internal and subject to change.

.. warning:: Mutations on the parameters of model evaluator (known as hyper-parameter tuning) is currently not supported but will be supported in the future.

.. warning:: To use PyTorch-lightning with Retiarii, currently you need to install PyTorch-lightning v1.1.x because v1.2 is not supported.

Detailed descriptions and usages can be found `here <./ApiReference.rst>`__ .

One-shot experiments
--------------------

One-shot is another family of popular NAS approaches, that does not require interactions between strategy and evaluator repeatedly, but combines everything into one model and one trainer. The trainer here takes charge of both search, training and testing.

We list the supported one-shot trainers here:

* DARTS trainer
* ENAS trainer
* ProxylessNAS trainer
* Single-path (random) trainer

See API reference for detailed usages. Here, we show an example to use DARTS trainer manually.

.. code-block:: python

  from nni.retiarii.oneshot.pytorch import DartsTrainer
  trainer = DartsTrainer(
      model=model,
      loss=criterion,
      metrics=lambda output, target: accuracy(output, target, topk=(1,)),
      optimizer=optim,
      num_epochs=args.epochs,
      dataset=dataset_train,
      batch_size=args.batch_size,
      log_frequency=args.log_frequency,
      unrolled=args.unrolled
  )
  trainer.fit()
  final_architecture = trainer.export()

Launch an Experiment
--------------------

After all the above are prepared, it is time to start an experiment to do the model search. We design unified interface for users to start their experiment. An example is shown below,

.. code-block:: python

  exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_strategy)
  exp_config = RetiariiExeConfig('local')
  exp_config.experiment_name = 'mnasnet_search'
  exp_config.trial_concurrency = 2
  exp_config.max_trial_number = 10
  exp_config.training_service.use_active_gpu = False
  exp.run(exp_config, 8081)

This code starts an NNI experiment. Note that if inlined mutation is used, ``applied_mutators`` should be ``None``.

For a one-shot experiment, it can be launched like this:

.. code-block:: python
  exp = RetiariiExperiment(base_model, oneshot_trainer)
  exp.run()

The complete code of a simple MNIST example can be found :githublink:`here <test/retiarii_test/mnist/test.py>`.

Visualize your experiment
-------------------------

Users can visualize their experiment in the same way as visualizing a normal hyper-parameter tuning experiment. For example, open ``localhost::8081`` in your browser, 8081 is the port that you set in ``exp.run``. Please refer to `here <../../Tutorial/WebUI.rst>`__ for details. If users are using oneshot trainer, they can refer to `here <../Visualization.rst>`__ for how to visualize their experiments.

Export the best model found in your experiment
----------------------------------------------

If you are using *classic search approach*, you can simply find out the best one from WebUI.

If you are using *oneshot (weight-sharing) search approach*, you can invole ``exp.export_top_models`` to output several best models that are found in the experiment.

Advanced
--------

Serializer
^^^^^^^^^^

.. _serializer:

The APIs like ``@basic_unit`` and ``serialize`` are all part of Retiarii as serializers. A serializer is basically needed for the following purposes:

* Prevent graph-parser to parse the module. To understand this, we first briefly explain how our framework works: it converts user-defined model to a graph representation (called graph IR), each instantiated module is converted to a subgraph. Then user-defined mutations are applied to the graph to generate new graphs. Each new graph is then converted back to PyTorch code and executed. ``@basic_unit`` here means the module will not be converted to a subgraph but is converted to a single graph node. That is, the module will not be unfolded anymore. When the subgraph is not unfolded, mutations on parameters of subgraph module becomes easier.
* Function as a translator to translate intermediate arguments (e.g., ValueChoice). A wrapper of module injects the chosen value and replaces the choice with the chosen one.
* Enable the re-instantiation of the object. Sometimes, modules and evaluators needs to be replicated and sent to training services. Retiarii needs to track how to instantiate them.

Thus, serializer should be used in the following cases:

* When a module class cannot be successfully converted to a subgraph due to some implementation issues. For example, currently our framework does not support adhoc loop, if there is adhoc loop in a module's forward, this class should be decorated as serializable module. The following ``MyModule`` should be decorated.

  .. code-block:: python

    @basic_unit
    class MyModule(nn.Module):
      def __init__(self):
        ...
      def forward(self, x):
        for i in range(10): # <- adhoc loop
          ...

* The candidate ops in ``LayerChoice`` should be decorated as serializable module. For example, ``self.op = nn.LayerChoice([Op1(...), Op2(...), Op3(...)])``, where ``Op1``, ``Op2``, ``Op3`` should be decorated if they are user defined modules.
* When users want to use ``ValueChoice`` in a module's input argument, the module should be decorated as serializable module. For example, ``self.conv = MyConv(kernel_size=nn.ValueChoice([1, 3, 5]))``, where ``MyConv`` should be decorated.
* If no mutation is targeted on a module, this module *can be* decorated as a serializable module.
