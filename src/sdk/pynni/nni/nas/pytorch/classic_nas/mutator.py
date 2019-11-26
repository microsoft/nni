import os
import sys
import json
import logging
import torch
import nni
from nni.env_vars import trial_env_vars
from nni.nas.pytorch.base_mutator import BaseMutator
from nni.nas.pytorch.mutables import LayerChoice, InputChoice

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

class ClassicMutator(BaseMutator):
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
        model : pytorch model
            user's model with search space (e.g., LayerChoice, InputChoice) embedded in it
        """
        super(ClassicMutator, self).__init__(model)
        self.chosen_arch = {}
        self.search_space = self._generate_search_space()
        if 'NNI_GEN_SEARCH_SPACE' in os.environ:
            # dry run for only generating search space
            self._dump_search_space(self.search_space, os.environ.get('NNI_GEN_SEARCH_SPACE'))
            sys.exit(0)
        # get chosen arch from tuner
        self.chosen_arch = nni.get_next_parameter()
        if not self.chosen_arch and trial_env_vars.NNI_PLATFORM is None:
            logger.warning('This is in standalone mode, the chosen are the first one(s)')
            self.chosen_arch = self._standalone_generate_chosen()
        self._validate_chosen_arch()

    def _validate_chosen_arch(self):
        pass

    def _standalone_generate_chosen(self):
        """
        Generate the chosen architecture for standalone mode,
        i.e., choose the first one(s) for LayerChoice and InputChoice

        { key_name: {'_value': "conv1",
                     '_idx': 0} }

        { key_name: {'_value': ["in1"],
                     '_idx': [0]} }

        Returns
        -------
        dict
            the chosen architecture
        """
        chosen_arch = {}
        for key, val in self.search_space.items():
            if val['_type'] == 'layer_choice':
                choices = val['_value']
                chosen_arch[key] = {'_value': choices[0], '_idx': 0}
            elif val['_type'] == 'input_choice':
                choices = val['_value']['candidates']
                n_chosen = val['_value']['n_chosen']
                chosen_arch[key] = {'_value': choices[:n_chosen], '_idx': list(range(n_chosen))}
            else:
                raise ValueError('Unknown key %s and value %s' % (key, val))
        return chosen_arch

    def _generate_search_space(self):
        """
        Generate search space from mutables.
        Here is the search space format:

        { key_name: {'_type': 'layer_choice',
                     '_value': ["conv1", "conv2"]} }

        { key_name: {'_type': 'input_choice',
                     '_value': {'candidates': ["in1", "in2"],
                                'n_chosen': 1}} }

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
                raise TypeError('Unsupported mutable type: %s.' % type(mutable))
        return search_space

    def _dump_search_space(self, search_space, file_path):
        with open(file_path, 'w') as ss_file:
            json.dump(search_space, ss_file)

    def _tensor_reduction(self, reduction_type, tensor_list):
        if tensor_list == "none":
            return tensor_list
        if not tensor_list:
            return None  # empty. return None for now
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == "sum":
            return sum(tensor_list)
        if reduction_type == "mean":
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == "concat":
            return torch.cat(tensor_list, dim=1)
        raise ValueError("Unrecognized reduction policy: \"{}\"".format(reduction_type))

    def on_forward_layer_choice(self, mutable, *inputs):
        """
        Implement the forward of LayerChoice

        Parameters
        ----------
        mutable: LayerChoice
        inputs: list of torch.Tensor

        Returns
        -------
        tuple
            return of the chosen op, the index of the chosen op

        """
        assert mutable.key in self.chosen_arch
        val = self.chosen_arch[mutable.key]
        assert isinstance(val, dict)
        idx = val['_idx']
        assert self.search_space[mutable.key]['_value'][idx] == val['_value']
        return mutable.choices[idx](*inputs), idx

    def on_forward_input_choice(self, mutable, tensor_list):
        """
        Implement the forward of InputChoice

        Parameters
        ----------
        mutable: InputChoice
        tensor_list: list of torch.Tensor
        tags: list of string

        Returns
        -------
        tuple of torch.Tensor and list
            reduced tensor, mask list

        """
        assert mutable.key in self.chosen_arch
        val = self.chosen_arch[mutable.key]
        assert isinstance(val, dict)
        mask = [0 for _ in range(mutable.n_candidates)]
        out = []
        for i, idx in enumerate(val['_idx']):
            # check whether idx matches the chosen candidate name
            assert self.search_space[mutable.key]['_value']['candidates'][idx] == val['_value'][i]
            out.append(tensor_list[idx])
            mask[idx] = 1
        return self._tensor_reduction(mutable.reduction, out), mask
