from copy import deepcopy
from typing import List, Dict, Callable, Optional

from torch.nn import Module

from .sparsity_generator import SparsityScheduler, SparsityAllocator, SparsityGenerator


class NaiveSparsityScheduler(SparsityScheduler):
    '''
    This scheduler control the sparsity by multiply a ratio related to iteration number.
    '''

    def __init__(self, config_list: List[Dict], iteration_num: Optional[int], schedule_type: str = 'constant'):
        '''
        Parameters
        ----------
        origin_config_list
            The original config list.
        iteration_num
            The iteration total number.
        schedule_type
            Support 'constant', 'linear' and 'agp'.
            'constant' means the ratio is 1.
            'linear' means the ratio is (iteration_round / iteration_num).
            'agp' means the ratio is (1 - (1 - iteration_round / iteration_num) ** 3).
        '''
        super().__init__(config_list, iteration_num=iteration_num)
        self._sparsity_schedule = self._get_sparsity_schedule(schedule_type)

    def get_next_config_list(self, real_sparsity_config_list: Optional[List[Dict]]) -> Optional[List[Dict]]:
        if self.iteration_num is not None:
            return self._limited_get_next_config_list()
        else:
            return self._unlimited_get_next_config_list()

    def _limited_get_next_config_list(self) -> Optional[List[Dict]]:
        if self.iteration_round >= self.iteration_num:
            return None

        self.iteration_round += 1
        config_list = deepcopy(self.origin_config_list)
        for config in config_list:
            if 'sparsity' in config:
                assert isinstance(config['sparsity'], float), 'Only support schedule float sparsity.'
                config['sparsity'] = self._sparsity_schedule(config['sparsity'])
        return config_list

    def _unlimited_get_next_config_list(self) -> Optional[List[Dict]]:
        return deepcopy(self.origin_config_list)

    def _get_sparsity_schedule(self, schedule_type: str) -> Callable[[float], float]:
        SCHEDULE_DICT = {
            'constant': self.__sparsity_schedule_constant,
            'linear': self.__sparsity_schedule_linear,
            'agp': self.__sparsity_schedule_agp
        }
        assert schedule_type in SCHEDULE_DICT, 'Unsupported schedule_type: {}'.format(schedule_type)
        return SCHEDULE_DICT[schedule_type]

    def __sparsity_schedule_constant(self, sparsity: float):
        return sparsity

    def __sparsity_schedule_linear(self, sparsity: float):
        ratio = self.iteration_round / self.iteration_num
        return sparsity * ratio

    def __sparsity_schedule_agp(self, sparsity: float):
        ratio = 1 - (1 - self.iteration_round / self.iteration_num) ** 3
        return sparsity * ratio


class NaiveSparsityAllocator(SparsityAllocator):
    '''
    Return the config_list directly.
    '''
    def get_allocated_config_list(self, model: Module, schedule_config_list: List[Dict]) -> List[Dict]:
        return schedule_config_list


class NaiveSparsityGenerrator(SparsityGenerator):
    def __init__(self, config_list: List[Dict], iteration_num: Optional[int], schedule_type: str = 'constant'):
        iteration_num = iteration_num if iteration_num else 1
        assert isinstance(iteration_num, int) and iteration_num > 0, 'iteration_num need greater than 0'
        self._sparsity_scheduler = NaiveSparsityScheduler(config_list, iteration_num, schedule_type)
        self._sparsity_allocator = NaiveSparsityAllocator()

    @property
    def sparsity_scheduler(self) -> SparsityScheduler:
        return self._sparsity_scheduler

    @property
    def sparsity_allocator(self) -> SparsityAllocator:
        return self._sparsity_allocator
