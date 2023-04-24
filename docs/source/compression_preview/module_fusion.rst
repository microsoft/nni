Module Fusion
=============

Module fusion is a new feature in the quantizatizer of NNI 3.0. This feature can fuse the specified
sub-models in the simulated quantization process to align with the inference stage of model deployment, 
reducing the error between the simulated quantization and inference stages.

Users can use this feature by directly defining ``fuse_names`` in each configure of config_list.
``fuse_names`` is an optional parameter of type ``List[(str,)]``. Each tuple specifies the name of the module 
to be fused in the current configure in the model. Meanwhile, each tuple has 2 or 3 elements, and the first module 
in each tuple is the fused module, which contains all the operations of all the modules in the tuple. 
The rest of the modules will be replaced by ``Identity`` during the quantization process. Here is an example:

.. code-block:: python

    # define the Mnist Model
    class Mnist(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
            self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
            self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
            self.fc2 = torch.nn.Linear(500, 10)
            self.relu1 = torch.nn.ReLU6()
            self.relu2 = torch.nn.ReLU6()
            self.relu3 = torch.nn.ReLU6()
            self.max_pool1 = torch.nn.MaxPool2d(2, 2)
            self.max_pool2 = torch.nn.MaxPool2d(2, 2)
            self.batchnorm1 = torch.nn.BatchNorm2d(20)

        def forward(self, x):
            x = self.relu1(self.batchnorm1(self.conv1(x)))
            x = self.max_pool1(x)
            x = self.relu2(self.conv2(x))
            x = self.max_pool2(x)
            x = x.view(-1, 4 * 4 * 50)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # define the config list
    config_list = [
    {
        'target_names':['_input_', 'weight', '_output_'],
        'op_names': ['conv1'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine',
        'granularity': 'default',
        'fuse_names': [("conv1", "batchnorm1")]
    }]


    