import torch.nn as nn


class Controller(nn.Module):
    """
    A controller is a sampler where an architecture is sampled, that is, a search/final decision is made,
    so that the decision can be understood by a general mutator and plugged directly into the network.
    A controller needs to be tuned to make better decisions, so it's exposed to trainers.
    """

    def build(self, mutables):
        """
        Once mutator has retrieved the structured representation of the search space, it will tell the controller
        to set up necessary parameters. This can't be combined into the `__init__` process because controllers are
        designed to be instantiated earlier than mutators, and they are meant to be isolated from the model itself,
        so when they are first initialized, the search space hasn't yet been obtained.
        In other words, this method serves as a "deferred initialization".

        Parameters
        ----------
        mutables: StructuredMutableTreeNode

        Returns
        -------
        None
        """
        raise NotImplementedError

    def sample_search(self, mutables):
        """
        Implement this method to iterate over mutables and make decisions.

        Parameters
        ----------
        mutables: StructuredMutableTreeNode

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError

    def sample_final(self, mutables):
        """
        Implement this method to iterate over mutables and make decisions that is final for export and retraining.

        Parameters
        ----------
        mutables: StructuredMutableTreeNode

        Returns
        -------
        dict
            A mapping from key of mutables to decisions.
        """
        raise NotImplementedError
