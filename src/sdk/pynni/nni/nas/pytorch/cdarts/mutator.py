# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from apex.parallel import DistributedDataParallel  # pylint: disable=import-error
from nni.nas.pytorch.darts import DartsMutator  # pylint: disable=wrong-import-order
from nni.nas.pytorch.mutables import LayerChoice  # pylint: disable=wrong-import-order
from nni.nas.pytorch.mutator import Mutator  # pylint: disable=wrong-import-order


class RegularizedDartsMutator(DartsMutator):
    """
    This is :class:`~nni.nas.pytorch.darts.DartsMutator` basically, with two differences.

    1. Choices can be cut (bypassed). This is done by ``cut_choices``. Cutted choices will not be used in
    forward pass and thus consumes no memory.

    2. Regularization on choices, to prevent the mutator from overfitting on some choices.
    """

    def reset(self):
        """
        Warnings
        --------
        Renamed :func:`~reset_with_loss` to return regularization loss on reset.
        """
        raise ValueError("You should probably call `reset_with_loss`.")

    def cut_choices(self, cut_num=2):
        """
        Cut the choices with the smallest weights.
        ``cut_num`` should be the accumulative number of cutting, e.g., if first time cutting
        is 2, the second time should be 4 to cut another two.

        Parameters
        ----------
        cut_num : int
            Number of choices to cut, so far.

        Warnings
        --------
        Though the parameters are set to :math:`-\infty` to be bypassed, they will still receive gradient of 0,
        which introduced ``nan`` problem when calling ``optimizer.step()``. To solve this issue, 
        """
        # `cut_choices` is implemented but not used in current implementation of CdartsTrainer
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                _, idx = torch.topk(-self.choices[mutable.key], cut_num)
                with torch.no_grad():
                    for i in idx:
                        self.choices[mutable.key][i] = -float("inf")

    def reset_with_loss(self):
        """
        Resample and return loss. If loss is 0, to avoid device issue, it will return ``None``.

        Currently loss penalty are proportional to the L1-norm of parameters corresponding
        to modules if their type name contains certain substrings. These substrings include: ``poolwithoutbn``,
        ``identity``, ``dilconv``.
        """
        self._cache, reg_loss = self.sample_search()
        return reg_loss

    def sample_search(self):
        result = super().sample_search()
        loss = []
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                def need_reg(choice):
                    return any(t in str(type(choice)).lower() for t in ["poolwithoutbn", "identity", "dilconv"])

                for i, choice in enumerate(mutable.choices):
                    if need_reg(choice):
                        norm = torch.abs(self.choices[mutable.key][i])
                        if norm < 1E10:
                            loss.append(norm)
        if not loss:
            return result, None
        return result, sum(loss)

    def export(self, logger=None):
        """
        Export an architecture with logger. Genotype will be printed with logger.

        Returns
        -------
        dict
            A mapping from mutable keys to decisions.
        """
        result = self.sample_final()
        if hasattr(self.model, "plot_genotype") and logger is not None:
            genotypes = self.model.plot_genotype(result, logger)
        return result, genotypes


class RegularizedMutatorParallel(DistributedDataParallel):
    """
    Parallelize :class:`~RegularizedDartsMutator`.

    This makes ``reset_with_loss`` method parallelized, also allowing ``cut_choies`` and
    ``export`` to be easily accessible (instead of using ``.module.``).
    """
    def reset_with_loss(self):
        result = self.module.reset_with_loss()
        self.callback_queued = False
        return result

    def cut_choices(self, *args, **kwargs):
        self.module.cut_choices(*args, **kwargs)

    def export(self, logger):
        return self.module.export(logger)


class DartsDiscreteMutator(Mutator):

    def __init__(self, model, parent_mutator):
        """
        Initialization.

        Parameters
        ----------
        model : nn.Module
            The model to apply the mutator.
        parent_mutator : Mutator
            The mutator that is used to call ``sample_final()`` method to get the architecture
            for training.
        """
        super().__init__(model)
        self.__dict__["parent_mutator"] = parent_mutator  # avoid parameters to be included

    def sample_search(self):
        return self.parent_mutator.sample_final()
