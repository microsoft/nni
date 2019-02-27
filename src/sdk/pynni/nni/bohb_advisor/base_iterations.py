import sys


import logging
import numpy as np
import pdb

class Job(object):
	def __init__(self, id, **kwargs):
		self.id = id
		
		self.kwargs = kwargs
		
		self.timestamps = {}

		self.result = None
		self.exception = None

		self.worker_name = None

	def time_it(self, which_time):
		self.timestamps[which_time] = time.time()

	def __repr__(self):
		return(\
			"job_id: " +str(self.id) + "\n" + \
			"kwargs: " + str(self.kwargs) + "\n" + \
			"result: " + str(self.result)+ "\n" +\
			"exception: "+ str(self.exception) + "\n"
		)
	def recreate_from_run(self, run):
		
		run.config_id
		run.budget
		run.error_logs  
		run.loss        
		run.info        
		run.time_stamps 

class Datum(object):
	def __init__(self, config, config_info, results=None, time_stamps=None, exceptions=None, status='QUEUED', budget=0):
        self.config        = config
        self.config_info= config_info
        self.results    = results	if not results is None else {}
		self.time_stamps= time_stamps	if not time_stamps is None else {}
		self.exceptions	= exceptions	if not exceptions is None else {}
		self.status		= status
		self.budget		= budget

	def __repr__(self):
		return(\
			"\nconfig:{}\n".format(self.config) + \
			"config_info:\n{}\n"%self.config_info + \
			"losses:\n"
			'\t'.join(["{}: {}\t".format(k, v['loss']) for k,v in self.results.items()]) + \
			"time stamps: {}".format(self.time_stamps)
		)



class BaseIteration(object):
	"""
	Base class for various iteration possibilities. This decides what configuration should be run on what budget
	next. Typical choices are e.g. successive halving. Results from runs are processed and (depending on the
	implementations) determine the further development.
	"""
	def __init__(self, HPB_iter, num_configs, budgets, config_sampler, logger=None, result_logger=None):
		"""
		Parameters
		----------

		HPB_iter: int
			The current HPBandSter iteration index.
		num_configs: list of ints
			the number of configurations in each stage of SH
		budgets: list of floats
			the budget associated with each stage
		config_sample: callable
			a function that returns a valid configuration. Its only
			argument should be the budget that this config is first
			scheduled for. This might be used to pick configurations
			that perform best after this particular budget is exhausted
			to build a better autoML system.
		logger: a logger
		result_logger: hpbandster.api.results.util.json_result_logger object
			a result logger that writes live results to disk
		"""

		self.data = {}					# this holds all the configs and results of this iteration
		self.is_finished = False
		self.HPB_iter = HPB_iter
		self.stage = 0					# internal iteration, but different name for clarity
		self.budgets = budgets
		self.num_configs = num_configs
		self.actual_num_configs = [0]*len(num_configs)
		self.config_sampler = config_sampler
		self.num_running = 0
		self.logger=logger if not logger is None else logging.getLogger('hpbandster')
		self.result_logger = result_logger

	def add_configuration(self, config = None, config_info={}):
		"""
		function to add a new configuration to the current iteration

		Parameters
		----------

		config : valid configuration
			The configuration to add. If None, a configuration is sampled from the config_sampler
		config_info: dict
			Some information about the configuration that will be stored in the results
		"""

		if config is None:
			config, config_info = self.config_sampler(self.budgets[self.stage])

		if self.is_finished:
			raise RuntimeError("This HPBandSter iteration is finished, you can't add more configurations!")

		if self.actual_num_configs[self.stage] == self.num_configs[self.stage]:
			raise RuntimeError("Can't add another configuration to stage %i in HPBandSter iteration %i."%(self.stage, self.HPB_iter))

		config_id = (self.HPB_iter, self.stage, self.actual_num_configs[self.stage])

		self.data[config_id] = Datum(config=config, config_info=config_info, budget = self.budgets[self.stage])

		self.actual_num_configs[self.stage] += 1

		if not self.result_logger is None:
		    self.result_logger.new_config(config_id, config, config_info)

		return(config_id)

	def register_result(self, job, skip_sanity_checks=False):
		"""
		function to register the result of a job

		This function is called from HB_master, don't call this from
		your script.
		"""

		if self.is_finished:
			raise RuntimeError("This HB iteration is finished, you can't register more results!")

		config_id = job.id
		config = job.kwargs['config']
		budget = job.kwargs['budget']
		timestamps = job.timestamps
		result = job.result
		exception = job.exception

		d = self.data[config_id]

		if not skip_sanity_checks:
			assert d.config == config, 'Configurations differ!'
			assert d.status == 'RUNNING', "Configuration wasn't scheduled for a run."
			assert d.budget == budget, 'Budgets differ (%f != %f)!'%(self.data[config_id]['budget'], budget)

		d.time_stamps[budget] = timestamps
		d.results[budget] = result

		if (not job.result is None) and np.isfinite(result['loss']):
			d.status = 'REVIEW'
		else:
			d.status = 'CRASHED'

		d.exceptions[budget] = exception
		self.num_running -= 1
		
	def get_next_run(self):
		"""
		function to return the next configuration and budget to run.

		This function is called from HB_master, don't call this from
		your script.

		It returns None if this run of SH is finished or there are
		pending jobs that need to finish to progress to the next stage.

		If there are empty slots to be filled in the current SH stage
		(which never happens in the original SH version), a new
		configuration will be sampled and scheduled to run next.
		"""

		if self.is_finished:
			return(None)
		
		for k,v in self.data.items():
			if v.status == 'QUEUED':
				assert v.budget == self.budgets[self.stage], 'Configuration budget does not align with current stage!'
				v.status = 'RUNNING'
				self.num_running += 1
				return(k, v.config, v.budget)

		# check if there are still slots to fill in the current stage and return that
		if (self.actual_num_configs[self.stage] < self.num_configs[self.stage]):
			self.add_configuration()
			return(self.get_next_run())

		if self.num_running == 0:
			# at this point a stage is completed
			self.process_results()
			return(self.get_next_run())

		return(None)

	def _advance_to_next_stage(self, config_ids, losses):
		"""
		Function that implements the strategy to advance configs within this iteration
		
		Overload this to implement different strategies, like
		SuccessiveHalving, SuccessiveResampling.
		
		Parameters
		----------
			config_ids: list
				all config ids to be considered
			losses: list
				losses of the run on the current budget
				
		Returns
		-------
			list of bool
				A boolean for each entry in config_ids indicating whether to advance it or not
		
		
		"""
		raise NotImplementedError('_advance_to_next_stage not implemented for %s'%type(self).__name__)

	def process_results(self):
		"""
		function that is called when a stage is completed and
		needs to be analyzed befor further computations.

		The code here implements the original SH algorithms by
		advancing the k-best (lowest loss) configurations at the current
		budget. k is defined by the num_configs list (see __init__)
		and the current stage value.

		For more advanced methods like resampling after each stage,
		overload this function only.
		"""
		self.stage += 1
		
		# collect all config_ids that need to be compared
		config_ids = list(filter(lambda cid: self.data[cid].status == 'REVIEW', self.data.keys()))

		if (self.stage >= len(self.num_configs)):
			self.finish_up()
			return

		budgets = [self.data[cid].budget for cid in config_ids]
		if len(set(budgets)) > 1:
			raise RuntimeError('Not all configurations have the same budget!')
		budget = self.budgets[self.stage-1]

		losses = np.array([self.data[cid].results[budget]['loss'] for cid in config_ids])

		advance = self._advance_to_next_stage(config_ids, losses)

		for i, a in enumerate(advance):
			if a:
				self.logger.debug('ITERATION: Advancing config %s to next budget %f'%(config_ids[i], self.budgets[self.stage]))

		for i, cid in enumerate(config_ids):
			if advance[i]:
				self.data[cid].status = 'QUEUED'
				self.data[cid].budget = self.budgets[self.stage]
				self.actual_num_configs[self.stage] += 1
			else:
				self.data[cid].status = 'TERMINATED'

	def finish_up(self):
		self.is_finished = True

		for k,v in self.data.items():
			assert v.status in ['TERMINATED', 'REVIEW', 'CRASHED'], 'Configuration has not finshed yet!'
			v.status = 'COMPLETED'


class WarmStartIteration(BaseIteration):
	"""
	iteration that imports a privious Result for warm starting
	"""

	def __init__(self, Result, config_generator):
		
		self.is_finished=False
		self.stage = 0

		
		id2conf = Result.get_id2config_mapping()
		delta_t = - max(map(lambda r: r.time_stamps['finished'], Result.get_all_runs()))

		super().__init__(-1, [len(id2conf)]	, [None], None)
		
		
		for i, id in enumerate(id2conf):
			new_id = self.add_configuration(config=id2conf[id]['config'], config_info=id2conf[id]['config_info'])
			
			for r in Result.get_runs_by_id(id):
			
				
				j = Job(new_id, config=id2conf[id]['config'], budget=r.budget)
				
				j.result = {'loss': r.loss, 'info': r.info}
				j.error_logs = r.error_logs
				
				for k,v in r.time_stamps.items():
					j.timestamps[k] = v + delta_t
				
				self.register_result(j , skip_sanity_checks=True)
				
				config_generator.new_result(j, update_model=(i==len(id2conf)-1))
				
		# mark as finished, as no more runs should be executed from these runs
		self.is_finished = True
		
	def fix_timestamps(self, time_ref):
		"""
			manipulates internal time stamps such that the last run ends at time 0
		"""
		
		for k,v in self.data.items():
			for kk, vv in v.time_stamps.items():
				for kkk,vvv in vv.items():
					self.data[k].time_stamps[kk][kkk] += time_ref

class SuccessiveHalving(BaseIteration):

	def _advance_to_next_stage(self, config_ids, losses):
		"""
			SuccessiveHalving simply continues the best based on the current loss.
		"""
		ranks = np.argsort(np.argsort(losses))
		return(ranks < self.num_configs[self.stage])
