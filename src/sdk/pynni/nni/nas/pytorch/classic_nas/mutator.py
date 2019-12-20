# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import sys

import torch

import nni
from nni.env_vars import trial_env_vars
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.nas.pytorch.mutator import Mutator

logger = logging.getLogger(__name__)


def get_and_apply_next_architecture(model):
    """
    Wrapper of ClassicMutator to make it more meaningful,
    similar to ```get_next_parameter``` for HPO.
    Parameters
    ----------
    model : pytorch model
        user's model with search space (e.g., LayerChoice, InputChoice) embedded in it
    """
    ClassicMutator(model)


class ClassicMutator(Mutator):
    """
    This mutator is to apply the architecture chosen from tuner.
    It implements the forward function of LayerChoice and InputChoice,
    to only activate the chosen ones
    """

    def __init__(self, model):
        """
        Generate search space based on ```model```.
        If env ```NNI_GEN_SEARCH_SPACE``` exists, this is in dry run mode for
        generating search space for the experiment.
        If not, there are still two mode, one is nni experiment mode where users
        use ```nnictl``` to start an experiment. The other is standalone mode
        where users directly run the trial command, this mode chooses the first
        one(s) for each LayerChoice and InputChoice.
        Parameters
        ----------
        model : PyTorch model
            user's model with search space (e.g., LayerChoice, InputChoice) embedded in it
        """
        super(ClassicMutator, self).__init__(model)
        self._chosen_arch = {}
        self._search_space = self._generate_search_space()
        if "NNI_GEN_SEARCH_SPACE" in os.environ:
            # dry run for only generating search space
            self._dump_search_space(os.environ["NNI_GEN_SEARCH_SPACE"])
            sys.exit(0)

        if trial_env_vars.NNI_PLATFORM is None:
            logger.warning("This is in standalone mode, the chosen are the first one(s).")
            self._chosen_arch = self._standalone_generate_chosen()
        else:
            # get chosen arch from tuner
            self._chosen_arch = nni.get_next_parameter()
        self._cache = self.sample_final()

    def sample_search(self):
        return self.sample_final()

    def sample_final(self):
        assert set(self._chosen_arch.keys()) == set(self._search_space.keys()), \
            "Unmatched keys, expected keys '{}' from search space, found '{}'.".format(self._search_space.keys(),
                                                                                       self._chosen_arch.keys())
        result = dict()
        for mutable in self.mutables:
            assert mutable.key in self._chosen_arch, "Expected '{}' in chosen arch, but not found.".format(mutable.key)
            data = self._chosen_arch[mutable.key]
            assert isinstance(data, dict) and "_value" in data and "_idx" in data, \
                "'{}' is not a valid choice.".format(data)
            value = data["_value"]
            idx = data["_idx"]
            search_space_ref = self._search_space[mutable.key]["_value"]
            if isinstance(mutable, LayerChoice):
                # doesn't support multihot for layer choice yet
                onehot_list = [False] * mutable.length
                assert 0 <= idx < mutable.length and search_space_ref[idx] == value, \
                    "Index '{}' in search space '{}' is not '{}'".format(idx, search_space_ref, value)
                onehot_list[idx] = True
                result[mutable.key] = torch.tensor(onehot_list, dtype=torch.bool)  # pylint: disable=not-callable
            elif isinstance(mutable, InputChoice):
                multihot_list = [False] * mutable.n_candidates
                for i, v in zip(idx, value):
                    assert 0 <= i < mutable.n_candidates and search_space_ref[i] == v, \
                        "Index '{}' in search space '{}' is not '{}'".format(i, search_space_ref, v)
                    assert not multihot_list[i], "'{}' is selected twice in '{}', which is not allowed.".format(i, idx)
                    multihot_list[i] = True
                result[mutable.key] = torch.tensor(multihot_list, dtype=torch.bool)  # pylint: disable=not-callable
            else:
                raise TypeError("Unsupported mutable type: '%s'." % type(mutable))
        return result

    def reset(self):
        pass  # do nothing, only sample once at initialization

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
            if val["_type"] == "layer_choice":
                choices = val["_value"]
                chosen_arch[key] = {"_value": choices[0], "_idx": 0}
            elif val["_type"] == "input_choice":
                choices = val["_value"]["candidates"]
                n_chosen = val["_value"]["n_chosen"]
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
                val = [repr(choice) for choice in mutable.choices]
                search_space[key] = {"_type": "layer_choice", "_value": val}
            elif isinstance(mutable, InputChoice):
                key = mutable.key
                search_space[key] = {"_type": "input_choice",
                                     "_value": {"candidates": mutable.choose_from,
                                                "n_chosen": mutable.n_chosen}}
            else:
                raise TypeError("Unsupported mutable type: '%s'." % type(mutable))
        return search_space

    def _dump_search_space(self, file_path):
        with open(file_path, "w") as ss_file:
            json.dump(self._search_space, ss_file, sort_keys=True, indent=2)
