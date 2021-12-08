# This file is to demo the usage of multi-trial NAS in the usage of SPOS search space.

import click
import json
import nni.retiarii.evaluator.pytorch as pl
import nni.retiarii.strategy as strategy
from nni.retiarii import serialize
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from torchvision import transforms
from torchvision.datasets import CIFAR10
from nn_meter import load_latency_predictor

from network import ShuffleNetV2OneShot
from utils import get_archchoice_by_model


class LatencyFilter:
    def __init__(self, threshold, predictor, predictor_version=None, reverse=False):
        """
        Filter the models according to predicted latency.

        Parameters
        ----------
        threshold: `float`
            the threshold of latency
        config, hardware:
            determine the targeted device
        reverse: `bool`
            if reverse is `False`, then the model returns `True` when `latency < threshold`,
            else otherwise
        """
        self.predictors = load_latency_predictor(predictor, predictor_version)
        self.threshold = threshold

    def __call__(self, ir_model):
        latency = self.predictors.predict(ir_model, 'nni-ir')
        return latency < self.threshold


@click.command()
@click.option('--port', default=8081, help='On which port the experiment is run.')
def _main(port):
    base_model = ShuffleNetV2OneShot(32)
    base_predictor = 'cortexA76cpu_tflite21'
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
    ]
    train_dataset = serialize(CIFAR10, 'data', train=True, download=True, transform=transforms.Compose(transf + normalize))
    test_dataset = serialize(CIFAR10, 'data', train=False, transform=transforms.Compose(normalize))

    trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=64),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=64),
                                max_epochs=2, gpus=1)

    simple_strategy = strategy.RegularizedEvolution(model_filter=LatencyFilter(threshold=100, predictor=base_predictor), population_size=2, cycles=2)
    exp = RetiariiExperiment(base_model, trainer, strategy=simple_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 2
    # exp_config.max_trial_number = 2
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = False
    exp_config.execution_engine = 'base'
    exp_config.dummy_input = [1, 3, 32, 32]

    exp.run(exp_config, port)

    print('Exported models:')
    for i, model in enumerate(exp.export_top_models(formatter='dict')):
        print(model)
        with open(f'architecture_final_{i}.json', 'w') as f: 
            json.dump(get_archchoice_by_model(model), f, indent=4)


if __name__ == '__main__':
    _main()
