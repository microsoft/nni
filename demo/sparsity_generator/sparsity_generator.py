from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from torch.nn import Module


# Various config_list are mentioned in the file.
# config_list: This kind of config list is the origin config list inputted by user.
# real_sparsity_config_list: This kind of config list describes the real sparsity rate of the last round.
# schedule_config_list: Generate from the config_list & real_sparsity_config_list for the next iteration globel config setting.
# allocated_config_list: A fine-grained config list generates from the schedule_config_list, used by pruner.

class SparsityScheduler(ABC):
    """
    This class used to schedule the sparsity for each iteration.
    """
    def __init__(self, config_list: List[Dict], iteration_num: Optional[int] = None):
        """
        Parameters
        ----------
        origin_config_list
            The original config list.
        iteration_num
            The iteration total number.
        """
        self.origin_config_list = config_list
        self.iteration_num = iteration_num
        self.iteration_round = 0

    @abstractmethod
    def get_next_config_list(self, real_sparsity_config_list: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """
        Parameters
        ----------
        real_sparsity_config_list
            The config list describe the real sparsity rate of the last round.

        Returns
        -------
        Optional[List[Dict]]
            The schedule config list for next iteration, None means no more iterations.
        """
        raise NotImplementedError()

    def reset(self, config_list: List[Dict], iteration_num: Optional[int] = None):
        """
        Reset the scheduler.

        Parameters
        ----------
        origin_config_list
            The original config list.
        iteration_num
            The iteration total number.
        """
        self.origin_config_list = config_list
        self.iteration_num = iteration_num
        self.iteration_round = 0


class SparsityAllocator(ABC):
    """
    This class used to allocate the sparsity for each layer in one iteration.
    """

    @abstractmethod
    def get_allocated_config_list(self, model: Module, schedule_config_list: List[Dict]) -> List[Dict]:
        """
        Parameters
        ----------
        model
            The model to be pruned.
        schedule_config_list
            The schedule config list used in this iteration.

        Returns
        -------
        List[Dict]
            The allocated config list for this iteration.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the allocator.
        """
        pass


class SparsityGenerator:
    """
    This class used to generate sparsity config list for pruner in each iteration.
    """
    @property
    @abstractmethod
    def sparsity_scheduler(self) -> SparsityScheduler:
        raise NotImplementedError()

    @property
    @abstractmethod
    def sparsity_allocator(self) -> SparsityAllocator:
        raise NotImplementedError()

    @property
    @abstractmethod
    def best_config_list(self) -> List[Dict]:
        raise NotImplementedError()

    def generate_config_list(self, model: Module, real_sparsity_config_list: List[Dict]) -> Optional[List[Dict]]:
        """
        Parameters
        ----------
        model
            The model that wants to sparsify.
        real_sparsity_config_list
            Real sparsity config list.

        Returns
        -------
        Optional[List[Dict]]
            The config list for this iteration, None means no more iterations.
        """
        schedule_config_list = self.sparsity_scheduler.get_next_config_list(real_sparsity_config_list)
        if schedule_config_list:
            allocated_config_list = self.sparsity_allocator.get_allocated_config_list(model, schedule_config_list)
            return allocated_config_list
        else:
            return None

    def reset(self, origin_config_list: List[Dict], iteration_num: Optional[int] = None):
        """
        Parameters
        ----------
        origin_config_list
            The origin config list.
        """
        self.sparsity_scheduler.reset(origin_config_list, iteration_num)
        self.sparsity_allocator.reset()
