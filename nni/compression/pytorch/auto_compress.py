import logging
from torch.nn import Module

_logger = logging.getLogger(__name__)


class AutoCompressor:
    """
    Auto Compression Class to utilize pruning & quantization in one place
    """

    def __init__(self) -> None:
        """
        Initialize required pruners and quantizers for automatic compressing

        Parameters
        ----------
        pruners : list<string>
            pruners chosen for compression
        quantizers : [list<string>, string]
            quantizers chosen for model speedup (can choose "All")
        training_dataloader : pytorch dataloader
            training dataset to use for model finetuning
        testing_dataloader : pytorch dataloader
            testing dataset to evaluate model
        evaluation_metric : pytorch functional
            metric to evaluate the compression result
        optimizer: pytorch optimizer
            optimizer used to train the model
        """
        pass

    def __call__(self) -> Module:
        """
        Compressing the model

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        verbose : int
            level of transparency in logging
        """
        pass