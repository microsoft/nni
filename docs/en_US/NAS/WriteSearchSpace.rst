Write A .. role:: raw-html(raw)
   :format: html

Search Space
====================

Genrally, a search space describes candiate architectures from which users want to find the best one. Different search algorithms, no matter classic NAS or one-shot NAS, can be applied on the search space. NNI provides APIs to unified the expression of neural architecture search space.

A search space can be built on a base model. This is also a common practice when a user wants to apply NAS on an existing model. Take `MNIST on PyTorch <https://github.com/pytorch/examples/blob/master/mnist/main.py>`__ as an example. Note that NNI provides the same APIs for expressing search space on PyTorch and TensorFlow.

.. code-block:: python

   from nni.nas.pytorch import mutables

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = mutables.LayerChoice([
               nn.Conv2d(1, 32, 3, 1),
               nn.Conv2d(1, 32, 5, 3)
           ])  # try 3x3 kernel and 5x5 kernel
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.dropout1 = nn.Dropout2d(0.25)
           self.dropout2 = nn.Dropout2d(0.5)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           # ... same as original ...
           return output

The example above adds an option of choosing conv5x5 at conv1. The modification is as simple as declaring a ``LayerChoice`` with the original conv3x3 and a new conv5x5 as its parameter. That's it! You don't have to modify the forward function in any way. You can imagine conv1 as any other module without NAS.

So how about the possibilities of connections? This can be done using ``InputChoice``. To allow for a skip connection on the MNIST example, we add another layer called conv3. In the following example, a possible connection from conv2 is added to the output of conv3.

.. code-block:: python

   from nni.nas.pytorch import mutables

   class Net(nn.Module):
       def __init__(self):
           # ... same ...
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.conv3 = nn.Conv2d(64, 64, 1, 1)
           # declaring that there is exactly one candidate to choose from
           # search strategy will choose one or None
           self.skipcon = mutables.InputChoice(n_candidates=1)
           # ... same ...

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           x = self.conv2(x)
           x0 = self.skipcon([x])  # choose one or none from [x]
           x = self.conv3(x)
           if x0 is not None:  # skipconnection is open
               x += x0
           x = F.max_pool2d(x, 2)
           # ... same ...
           return output

Input choice can be thought of as a callable module that receives a list of tensors and outputs the concatenation/sum/mean of some of them (sum by default), or ``None`` if none is selected. Like layer choices, input choices should be **initialized in ``__init__`` and called in ``forward``**. This is to allow search algorithms to identify these choices and do necessary preparations.

``LayerChoice`` and ``InputChoice`` are both **mutables**. Mutable means "changeable". As opposed to traditional deep learning layers/modules which have fixed operation types once defined, models with mutable are essentially a series of possible models.

Users can specify a **key** for each mutable. By default, NNI will assign one for you that is globally unique, but in case users want to share choices (for example, there are two ``LayerChoice``\ s with the same candidate operations and you want them to have the same choice, i.e., if first one chooses the i-th op, the second one also chooses the i-th op), they can give them the same key. The key marks the identity for this choice and will be used in the dumped checkpoint. So if you want to increase the readability of your exported architecture, manually assigning keys to each mutable would be a good idea. For advanced usage on mutables (e.g., ``LayerChoice`` and ``InputChoice``\ ), see `Mutables <./NasReference.rst>`__.

With search space defined, the next step is searching for the best model from it. Please refer to `classic NAS algorithms <./ClassicNas.md>`__ and `one-shot NAS algorithms <./NasGuide.rst>`__ for how to search from your defined search space.
