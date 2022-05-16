"""
Nas Example: One-shot DARTS
===========================
This is an end-to-end example for DARTS. In this tutorial, you will learn how to implement architecture search via DARTS algorthms,
and retrain the model based on the best architecture. You can read more about the DARTS in the [DARTS paper](https://arxiv.org/abs/1806.09055).

[DARTS](https://github.com/quark0/darts) addresses the scalability challenge of architecture search by formulating the task in a differentiable manner.
Their method is based on the continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent.

The code in this example is implemented by NNI based on the [official implementation](https://github.com/quark0/darts) and a
[popular 3rd-party repo](https://github.com/khanrc/pt.darts). DARTS on NNI is designed to be general for arbitrary search space and arbitrary dataset.
In this use case, a CNN search space tailored for CIFAR10, is implemented to synchronize with the original paper.

Loading the data
----------------

In this post we experiment with CIFAR10 dataset. The dataset will be downloaded into `./data/cifar-10-batches-py` if there is no local dataset.
"""
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import RandomSampler
from nni.retiarii.evaluator.pytorch.lightning import DataLoader

MEAN = [0.49139968, 0.48215827, 0.44653124]
STD = [0.24703233, 0.24348505, 0.26158768]
transf = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()
]
normalize = [
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
]
train_transform = transforms.Compose(transf + normalize)
valid_transform = transforms.Compose(normalize)

dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
train_random_sampler = RandomSampler(dataset_train, True, int(len(dataset_train) / 10))
train_loader = DataLoader(dataset_train, 64, sampler = train_random_sampler)

dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
valid_random_sampler = RandomSampler(dataset_valid, True, int(len(dataset_valid) / 10))
valid_loader = DataLoader(dataset_valid, 64, sampler = valid_random_sampler)


# %%
#
# Show search space
# -----------------
#
# We utilize a CNN search space, which is implemented in `./model.py`.
# insert image
# Figure 1: An overview of DARTS: (a) Operations on the edges are initially unknown. (b) Continuous
# relaxation of the search space by placing a mixture of candidate operations on each edge. (c) Joint
# optimization of the mixing probabilities and the network weights by solving a bilevel optimization
# problem. (d) Inducing the final architecture from the learned mixing probabilities.

# TODO: show the architecture

from model import CNN

model = CNN(32, 3, channels=16, n_classes=10, n_layers=8)

# %%
#
# Searching the best architecture
# -------------------------------
# Firstly we define the training module for architecture search. NNI apply the interface of
# ``nni.retiarii.experiment.pytorch.RetiariiExperiment`` to set up a NAS experiment. Users should specify the base model, evaluator, and strategy
# to start the experiment. In this project, we set the evaluator as a classification module, and set the exploration strategy to DARTS. In addition,
# Configurations of experiments, such as execution engine should be specified as :class:`RetiariiExeConfig`. The overall code is:

from nni.retiarii.strategy import DARTS
from nni.retiarii.evaluator.pytorch.lightning import Classification
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment

cls = Classification(train_dataloader=train_loader, val_dataloaders=valid_loader, max_epochs=1)
strategy_ = DARTS()
experiment = RetiariiExperiment(model, cls, strategy=strategy_)

config = RetiariiExeConfig()
config.execution_engine = 'oneshot'

# %%
#
# Now, let’s use the DARTS strategy to train the model. The network is based on continuous relaxation and gradient descent in the architecture space.
# In training, the network optimizes the network weights and architecture weights alternatively in an end-to-end fashion. Users can start the training by:

experiment.run(config)


# %%
# Visualizing the results and export model
# ----------------------------------------
#
# After searching in the CNN search space. The best architecture in status will be stored in `trainer` and can be exported via:

final_architecture = experiment.export_top_models(formatter='dict')
print('Final architecture:', experiment.export())

# dump best architecture by json
import json
json.dump(final_architecture, open('checkpoint.json', 'w'))

# %%
#
# Retrain the model with searched architecture
# --------------------------------------------
# Now we have a final architecture in the previous step, we can also evaluate our best architectures by training from scratch. To load the best architecture in searching, run:

from nni.retiarii import fixed_arch

# Load architecture from ``fixed_arch`` and apply to model
with fixed_arch('checkpoint.json'):
    model = CNN(32, 3, channels=16, n_classes=10, n_layers=8)
# import pdb; pdb.set_trace()

# %%
#
# Load evaluator and run the evaluation:

cls = Classification(train_dataloader=train_loader, val_dataloaders=valid_loader, max_epochs=1)
cls.fit(model)

# %%
#
# Performance of DARTS in NNI
# ---------------------------
# run results and document them in docs

# Todo List
# - [doing] test nas experiment in searching and evaluation
# - [done] visulization metrics via tensorboard
# - [blocked] checkpoint: model weights, architecture
# - [blocked] monitor logs via tensorboard
# - [blocked] try diverse search spaces to test DARTS algorithm
# - [blocked] need redesign after refactor with lightning
#     - utilize hardware-aware metric
#     - architecture checkpoint hook
