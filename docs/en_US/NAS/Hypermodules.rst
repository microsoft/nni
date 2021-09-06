Hypermodules
============

Hypermodule is a (PyTorch) module which contains many architecture/hyperparameter candidates for this module. By using hypermodule in user defined model, NNI will help users automatically find the best architecture/hyperparameter of the hypermodules for this model. This follows the design philosophy of Retiarii that users write DNN model as a space.

There has been proposed some hypermodules in NAS community, such as AutoActivation, AutoDropout. Some of them are implemented in the Retiarii framework.

..  autoclass:: nni.retiarii.nn.pytorch.AutoActivation
    :members: