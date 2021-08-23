# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import sys

import tensorflow as tf

import nni
from nni.runtime.env_vars import trial_env_vars
from nni.nas.tensorflow.mutables import LayerChoice, InputChoice, MutableScope
from nni.nas.tensorflow.mutator import Mutator

logger = logging.getLogger(__name__)

NNI_GEN_SEARCH_SPACE = "NNI_GEN_SEARCH_SPACE"
LAYER_CHOICE = "layer_choice"
INPUT_CHOICE = "input_choice"


def get_and_apply_next_architecture(model):
    """
    Wrapper of :class:`~nni.nas.tensorflow.classic_nas.mutator.ClassicMutator` to make it more meaningful,
    similar to ``get_next_parameter`` for HPO.
    Tt will generate search space based on ``model``.
    If env ``NNI_GEN_SEARCH_SPACE`` exists, this is in dry run mode for
    generating search space for the experiment.
    If not, there are still two mode, one is nni experiment mode where users
    use ``nnictl`` to start an experiment. The other is standalone mode
    where users directly run the trial command, this mode chooses the first
    one(s) for each LayerChoice and InputChoice.
    Parameters
    ----------
    model : nn.Module
        User's model with search space (e.g., LayerChoice, InputChoice) embedded in it.
    """
    ClassicMutator(model)


class ClassicMutator(Mutator):
    """
    This mutator is to apply the architecture chosen from tuner.
    It implements the forward function of LayerChoice and InputChoice,
    to only activate the chosen ones.
    Parameters
    ----------
    model : nn.Module
        User's model with search space (e.g., LayerChoice, InputChoice) embedded in it.
    """

    def __init__(self, model):
        super(ClassicMutator, self).__init__(model)
        self._chosen_arch = {}
        self._search_space = self._generate_search_space()
        if NNI_GEN_SEARCH_SPACE in os.environ:
            # dry run for only generating search space
            self._dump_search_space(os.environ[NNI_GEN_SEARCH_SPACE])
            sys.exit(0)

        if trial_env_vars.NNI_PLATFORM is None:
            logger.warning("This is in standalone mode, the chosen are the first one(s).")
            self._chosen_arch = self._standalone_generate_chosen()
        else:
            # get chosen arch from tuner
            self._chosen_arch = nni.get_next_parameter()
            if self._chosen_arch is None:
                if trial_env_vars.NNI_PLATFORM == "unittest":
                    # happens if NNI_PLATFORM is intentionally set, e.g., in UT
                    logger.warning("`NNI_PLATFORM` is set but `param` is None. Falling back to standalone mode.")
                    self._chosen_arch = self._standalone_generate_chosen()
                else:
                    raise RuntimeError("Chosen architecture is None. This may be a platform error.")
        self.reset()

    def _sample_layer_choice(self, mutable, idx, value, search_space_item):
        """
        Convert layer choice to tensor representation.
        Parameters
        ----------
        mutable : Mutable
        idx : int
            Number `idx` of list will be selected.
        value : str
            The verbose representation of the selected value.
        search_space_item : list
            The list for corresponding search space.
        """
        # doesn't support multihot for layer choice yet
        assert 0 <= idx < len(mutable) and search_space_item[idx] == value, \
            "Index '{}' in search space '{}' is not '{}'".format(idx, search_space_item, value)
        mask = tf.one_hot(idx, len(mutable))
        return tf.cast(tf.reshape(mask, [-1]), tf.bool)

    def _sample_input_choice(self, mutable, idx, value, search_space_item):
        """
        Convert input choice to tensor representation.
        Parameters
        ----------
        mutable : Mutable
        idx : int
            Number `idx` of list will be selected.
        value : str
            The verbose representation of the selected value.
        search_space_item : list
            The list for corresponding search space.
        """
        candidate_repr = search_space_item["candidates"]
        multihot_list = [False] * mutable.n_candidates
        for i, v in zip(idx, value):
            assert 0 <= i < mutable.n_candidates and candidate_repr[i] == v, \
                "Index '{}' in search space '{}' is not '{}'".format(i, candidate_repr, v)
            assert not multihot_list[i], "'{}' is selected twice in '{}', which is not allowed.".format(i, idx)
            multihot_list[i] = True
        return tf.cast(multihot_list, tf.bool)  # pylint: disable=not-callable

    def sample_search(self):
        """
        See :meth:`sample_final`.
        """
        return self.sample_final()

    def sample_final(self):
        """
        Convert the chosen arch and apply it on model.
        """
        assert set(self._chosen_arch.keys()) == set(self._search_space.keys()), \
            "Unmatched keys, expected keys '{}' from search space, found '{}'.".format(self._search_space.keys(),
                                                                                       self._chosen_arch.keys())
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, (LayerChoice, InputChoice)):
                assert mutable.key in self._chosen_arch, \
                    "Expected '{}' in chosen arch, but not found.".format(mutable.key)
                data = self._chosen_arch[mutable.key]
                assert isinstance(data, dict) and "_value" in data and "_idx" in data, \
                    "'{}' is not a valid choice.".format(data)
            if isinstance(mutable, LayerChoice):
                result[mutable.key] = self._sample_layer_choice(mutable, data["_idx"], data["_value"],
                                                                self._search_space[mutable.key]["_value"])
            elif isinstance(mutable, InputChoice):
                result[mutable.key] = self._sample_input_choice(mutable, data["_idx"], data["_value"],
                                                                self._search_space[mutable.key]["_value"])
            elif isinstance(mutable, MutableScope):
                logger.info("Mutable scope '%s' is skipped during parsing choices.", mutable.key)
            else:
                raise TypeError("Unsupported mutable type: '%s'." % type(mutable))
        return result

    def _standalone_generate_chosen(self):
        """
        Generate the chosen architecture for standalone mode,
        i.e., choose the first one(s) for LayerChoice and InputChoice.
        ::
            { key_name: {"_value": "conv1",
                         "_idx": 0} }
            { key_name: {"_value": ["in1"],
                         "_idx": [0]} }
        Returns
        -------
        dict
            the chosen architecture
        """
        chosen_arch = {}
        for key, val in self._search_space.items():
            if val["_type"] == LAYER_CHOICE:
                choices = val["_value"]
                chosen_arch[key] = {"_value": choices[0], "_idx": 0}
            elif val["_type"] == INPUT_CHOICE:
                choices = val["_value"]["candidates"]
                n_chosen = val["_value"]["n_chosen"]
                if n_chosen is None:
                    n_chosen = len(choices)
                chosen_arch[key] = {"_value": choices[:n_chosen], "_idx": list(range(n_chosen))}
            else:
                raise ValueError("Unknown key '%s' and value '%s'." % (key, val))
        return chosen_arch

    def _generate_search_space(self):
        """
        Generate search space from mutables.
        Here is the search space format:
        ::
            { key_name: {"_type": "layer_choice",
                         "_value": ["conv1", "conv2"]} }
            { key_name: {"_type": "input_choice",
                         "_value": {"candidates": ["in1", "in2"],
                                    "n_chosen": 1}} }
        Returns
        -------
        dict
            the generated search space
        """
        search_space = {}
        for mutable in self.mutables:
            # for now we only generate flattened search space
            if isinstance(mutable, LayerChoice):
                key = mutable.key
                val = mutable.names
                search_space[key] = {"_type": LAYER_CHOICE, "_value": val}
            elif isinstance(mutable, InputChoice):
                key = mutable.key
                search_space[key] = {"_type": INPUT_CHOICE,
                                     "_value": {"candidates": mutable.choose_from,
                                                "n_chosen": mutable.n_chosen}}
            elif isinstance(mutable, MutableScope):
                logger.info("Mutable scope '%s' is skipped during generating search space.", mutable.key)
            else:
                raise TypeError("Unsupported mutable type: '%s'." % type(mutable))
        return search_space

    def _dump_search_space(self, file_path):
        with open(file_path, "w") as ss_file:
            json.dump(self._search_space, ss_file, sort_keys=True, indent=2)
