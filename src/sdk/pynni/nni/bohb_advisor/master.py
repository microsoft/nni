import os
import threading
import time
import math
import pdb
import copy
import logging

import numpy as np


from .dispatcher import Dispatcher
from .result import Result
from .base_iteration import WarmStartIteration




class Master(object):
	def __init__(self,
			run_id,
			config_generator,
			working_directory='.',
			ping_interval=60,
			nameserver='127.0.0.1',
			nameserver_port=None,
			host=None,
			shutdown_workers=True,
			job_queue_sizes=(-1,0),
			dynamic_queue_size=True,
			logger=None,
			result_logger=None,
			previous_result = None,
			):
		"""The Master class is responsible for the book keeping and to decide what to run next. Optimizers are
                instantiations of Master, that handle the important steps of deciding what configurations to run on what
                budget when.
		
		Parameters
		----------
		run_id : string
			A unique identifier of that Hyperband run. Use, for example, the cluster's JobID when running multiple
			concurrent runs to separate them
		config_generator: hpbandster.config_generators object
			An object that can generate new configurations and registers results of executed runs
		working_directory: string
			The top level working directory accessible to all compute nodes(shared filesystem).
		eta : float
			In each iteration, a complete run of sequential halving is executed. In it,
			after evaluating each configuration on the same subset size, only a fraction of
			1/eta of them 'advances' to the next round.
			Must be greater or equal to 2.
		min_budget : float
			The smallest budget to consider. Needs to be positive!
		max_budget : float
			the largest budget to consider. Needs to be larger than min_budget!
			The budgets will be geometrically distributed :math:`\sim \eta^k` for
			:math:`k\in [0, 1, ... , num\_subsets - 1]`.
		ping_interval: int
			number of seconds between pings to discover new nodes. Default is 60 seconds.
		nameserver: str
			address of the Pyro4 nameserver
		nameserver_port: int
			port of Pyro4 nameserver
		host: str
			ip (or name that resolves to that) of the network interface to use
		shutdown_workers: bool
			flag to control whether the workers are shutdown after the computation is done
		job_queue_size: tuple of ints
			min and max size of the job queue. During the run, when the number of jobs in the queue
			reaches the min value, it will be filled up to the max size. Default: (0,1)
		dynamic_queue_size: bool
			Whether or not to change the queue size based on the number of workers available.
			If true (default), the job_queue_sizes are relative to the current number of workers.
		logger: logging.logger like object
			the logger to output some (more or less meaningful) information
		result_logger: hpbandster.api.results.util.json_result_logger object
			a result logger that writes live results to disk
		previous_result: hpbandster.core.result.Result object
			previous run to warmstart the run
		"""

		self.working_directory = working_directory
		os.makedirs(self.working_directory, exist_ok=True)
		

		if logger is None:
			self.logger = logging.getLogger('hpbandster')
		else:
			self.logger = logger

		self.result_logger = result_logger


		self.config_generator = config_generator
		self.time_ref = None


		self.iterations = []
		self.jobs = []

		self.num_running_jobs = 0
		self.job_queue_sizes = job_queue_sizes
		self.user_job_queue_sizes = job_queue_sizes
		self.dynamic_queue_size = dynamic_queue_size

		if job_queue_sizes[0] >= job_queue_sizes[1]:
			raise ValueError("The queue size range needs to be (min, max) with min<max!")

		if previous_result is None:
			self.warmstart_iteration = []

		else:
			self.warmstart_iteration = [WarmStartIteration(previous_result, self.config_generator)]

		# condition to synchronize the job_callback and the queue
		self.thread_cond = threading.Condition()

		self.config = {
						'time_ref'   : self.time_ref
					}

		self.dispatcher = Dispatcher( self.job_callback, queue_callback=self.adjust_queue_size, run_id=run_id, ping_interval=ping_interval, nameserver=nameserver, nameserver_port=nameserver_port, host=host)

		self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
		self.dispatcher_thread.start()


	def shutdown(self, shutdown_workers=False):
		self.logger.debug('HBMASTER: shutdown initiated, shutdown_workers = %s'%(str(shutdown_workers)))
		self.dispatcher.shutdown(shutdown_workers)
		self.dispatcher_thread.join()


	def wait_for_workers(self, min_n_workers=1):
		"""
		helper function to hold execution until some workers are active

		Parameters
		----------
		min_n_workers: int
			minimum number of workers present before the run starts		
		"""
	
		self.logger.debug('wait_for_workers trying to get the condition')
		with self.thread_cond:
			while (self.dispatcher.number_of_workers() < min_n_workers):
				self.logger.debug('HBMASTER: only %i worker(s) available, waiting for at least %i.'%(self.dispatcher.number_of_workers(), min_n_workers))
				self.thread_cond.wait(1)
				self.dispatcher.trigger_discover_worker()
				
		self.logger.debug('Enough workers to start this run!')
			

	def get_next_iteration(self, iteration, iteration_kwargs):
		"""
		instantiates the next iteration

		Overwrite this to change the iterations for different optimizers

		Parameters
		----------
			iteration: int
				the index of the iteration to be instantiated
			iteration_kwargs: dict
				additional kwargs for the iteration class

		Returns
		-------
			HB_iteration: a valid HB iteration object
		"""
		
		raise NotImplementedError('implement get_next_iteration for %s'%(type(self).__name__))


	def run(self, n_iterations=1, min_n_workers=1, iteration_kwargs = {},):
		"""
			run n_iterations of SuccessiveHalving

		Parameters
		----------
		n_iterations: int
			number of iterations to be performed in this run
		min_n_workers: int
			minimum number of workers before starting the run
		"""

		self.wait_for_workers(min_n_workers)
		
		iteration_kwargs.update({'result_logger': self.result_logger})

		if self.time_ref is None:
			self.time_ref = time.time()
			self.config['time_ref'] = self.time_ref
		
			self.logger.info('HBMASTER: starting run at %s'%(str(self.time_ref)))

		self.thread_cond.acquire()
		while True:

			self._queue_wait()
			
			next_run = None
			# find a new run to schedule
			for i in self.active_iterations():
				next_run = self.iterations[i].get_next_run()
				if not next_run is None: break

			if not next_run is None:
				self.logger.debug('HBMASTER: schedule new run for iteration %i'%i)
				self._submit_job(*next_run)
				continue
			else:
				if n_iterations > 0:	#we might be able to start the next iteration
					self.iterations.append(self.get_next_iteration(len(self.iterations), iteration_kwargs))
					n_iterations -= 1
					continue

			# at this point there is no imediate run that can be scheduled,
			# so wait for some job to finish if there are active iterations
			if self.active_iterations():
				self.thread_cond.wait()
			else:
				break

		self.thread_cond.release()
		
		for i in self.warmstart_iteration:
			i.fix_timestamps(self.time_ref)
			
		ws_data = [i.data for i in self.warmstart_iteration]
		
		return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data, self.config)


	def adjust_queue_size(self, number_of_workers=None):

		self.logger.debug('HBMASTER: number of workers changed to %s'%str(number_of_workers))
		with self.thread_cond:
			self.logger.debug('adjust_queue_size: lock accquired')
			if self.dynamic_queue_size:
				nw = self.dispatcher.number_of_workers() if number_of_workers is None else number_of_workers
				self.job_queue_sizes = (self.user_job_queue_sizes[0] + nw, self.user_job_queue_sizes[1] + nw)
				self.logger.info('HBMASTER: adjusted queue size to %s'%str(self.job_queue_sizes))
			self.thread_cond.notify_all()


	def job_callback(self, job):
		"""
		method to be called when a job has finished

		this will do some book keeping and call the user defined
		new_result_callback if one was specified
		"""
		self.logger.debug('job_callback for %s started'%str(job.id))
		with self.thread_cond:
			self.logger.debug('job_callback for %s got condition'%str(job.id))
			self.num_running_jobs -= 1

			if not self.result_logger is None:
				self.result_logger(job)
			self.iterations[job.id[0]].register_result(job)
			self.config_generator.new_result(job)

			if self.num_running_jobs <= self.job_queue_sizes[0]:
				self.logger.debug("HBMASTER: Trying to run another job!")
				self.thread_cond.notify()

		self.logger.debug('job_callback for %s finished'%str(job.id))

	def _queue_wait(self):
		"""
		helper function to wait for the queue to not overflow/underload it
		"""
		
		if self.num_running_jobs >= self.job_queue_sizes[1]:
			while(self.num_running_jobs > self.job_queue_sizes[0]):
				self.logger.debug('HBMASTER: running jobs: %i, queue sizes: %s -> wait'%(self.num_running_jobs, str(self.job_queue_sizes)))
				self.thread_cond.wait()

	def _submit_job(self, config_id, config, budget):
		"""
		hidden function to submit a new job to the dispatcher

		This function handles the actual submission in a
		(hopefully) thread save way
		"""
		self.logger.debug('HBMASTER: trying submitting job %s to dispatcher'%str(config_id))
		with self.thread_cond:
			self.logger.debug('HBMASTER: submitting job %s to dispatcher'%str(config_id))
			self.dispatcher.submit_job(config_id, config=config, budget=budget, working_directory=self.working_directory)
			self.num_running_jobs += 1

		#shouldn't the next line be executed while holding the condition?
		self.logger.debug("HBMASTER: job %s submitted to dispatcher"%str(config_id))

	def active_iterations(self):
		"""
		function to find active (not marked as finished) iterations 

		Returns
		-------
			list: all active iteration objects (empty if there are none)
		"""

		l = list(filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations))))
		return(l)

	def __del__(self):
		pass
