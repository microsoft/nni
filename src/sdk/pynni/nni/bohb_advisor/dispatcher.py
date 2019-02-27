import threading
import logging
import queue
import time

import Pyro4


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




class Worker(object):
	def __init__(self, name, uri):
		self.name = name
		self.proxy = Pyro4.Proxy(uri)
		self.runs_job = None

	def is_alive(self):
		try:
			self.proxy._pyroReconnect(1)
		except Pyro4.errors.ConnectionClosedError:
			return False
		except:
			raise
		return(True)
	
	def shutdown(self):
		self.proxy.shutdown()

	def is_busy(self):
		return(self.proxy.is_busy())
		
	def __repr__(self):
		return(self.name)


class Dispatcher(object):
	"""
	The dispatcher is responsible for assigning tasks to free workers, report results back to the master and
	communicate to the nameserver.
	"""
	def __init__(self, new_result_callback, run_id='0',
					ping_interval=10, nameserver='localhost',
					nameserver_port=None, 
					host=None, logger=None, queue_callback=None):
		"""
		Parameters
		----------
		new_result_callback: function
		    function that will be called with a `Job instance <hpbandster.core.dispatcher.Job>`_ as argument.
		    From the `Job` the result can be read and e.g. logged.
		run_id: str
		    unique run_id associated with the HPB run
		ping_interval: int
		    how often to ping for workers (in seconds)
		nameserver: str
		    address of the Pyro4 nameserver
		nameserver_port: int
		    port of Pyro4 nameserver
		host: str
		    ip (or name that resolves to that) of the network interface to use
		logger: logging.Logger
		    logger-instance for info and debug
		queue_callback: function
		    gets called with the number of workers in the pool on every update-cycle
		"""

		self.new_result_callback = new_result_callback
		self.queue_callback = queue_callback
		self.run_id = run_id
		self.nameserver = nameserver
		self.nameserver_port = nameserver_port
		self.host = host
		self.ping_interval = int(ping_interval)
		self.shutdown_all_threads = False


		if logger is None:
			self.logger = logging.getLogger('hpbandster')
		else:
			self.logger = logger

		self.worker_pool = {}

		self.waiting_jobs = queue.Queue()
		self.running_jobs = {}
		self.idle_workers = set()


		self.thread_lock = threading.Lock()
		self.runner_cond = threading.Condition(self.thread_lock)
		self.discover_cond = threading.Condition(self.thread_lock)

		self.pyro_id="hpbandster.run_%s.dispatcher"%self.run_id


	def run(self):
		with self.discover_cond:
			t1 = threading.Thread(target=self.discover_workers, name='discover_workers')
			t1.start()
			self.logger.info('DISPATCHER: started the \'discover_worker\' thread')
			t2 = threading.Thread(target=self.job_runner, name='job_runner')
			t2.start()
			self.logger.info('DISPATCHER: started the \'job_runner\' thread')
	

			self.pyro_daemon = Pyro4.core.Daemon(host=self.host)

			with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
				uri = self.pyro_daemon.register(self, self.pyro_id)
				ns.register(self.pyro_id, uri)

			self.logger.info("DISPATCHER: Pyro daemon running on %s"%(self.pyro_daemon.locationStr))

		
		self.pyro_daemon.requestLoop()


		with self.discover_cond:
			self.shutdown_all_threads = True
			self.logger.info('DISPATCHER: Dispatcher shutting down')
			
			self.runner_cond.notify_all()
			self.discover_cond.notify_all()
			
			
		
			with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
				ns.remove(self.pyro_id)

		t1.join()
		self.logger.debug('DISPATCHER: \'discover_worker\' thread exited')
		t2.join()
		self.logger.debug('DISPATCHER: \'job_runner\' thread exited')
		self.logger.info('DISPATCHER: shut down complete')

	def shutdown_all_workers(self, rediscover=False):
		with self.discover_cond:
			for worker in self.worker_pool.values():
				worker.shutdown()
			if rediscover:
				time.sleep(1)
				self.discover_cond.notify()

	def shutdown(self, shutdown_workers=False):
		if shutdown_workers:
			self.shutdown_all_workers()

		with self.runner_cond:
			self.pyro_daemon.shutdown()
	
	@Pyro4.expose
	@Pyro4.oneway
	def trigger_discover_worker(self):
		#time.sleep(1)
		self.logger.info("DISPATCHER: A new worker triggered discover_worker")
		with self.discover_cond:
			self.discover_cond.notify()

	
	def discover_workers(self):
		self.discover_cond.acquire()
		sleep_interval = 1
		
		while True:
			self.logger.debug('DISPATCHER: Starting worker discovery')
			update = False
		
			with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
				worker_names = ns.list(prefix="hpbandster.run_%s.worker."%self.run_id)
				self.logger.debug("DISPATCHER: Found %i potential workers, %i currently in the pool."%(len(worker_names), len(self.worker_pool)))
				
				for wn, uri in worker_names.items():
					if not wn in self.worker_pool:
						w = Worker(wn, uri)
						if not w.is_alive():
							self.logger.debug('DISPATCHER: skipping dead worker, %s'%wn)
							continue 
						update = True
						self.logger.info('DISPATCHER: discovered new worker, %s'%wn)
						self.worker_pool[wn] = w

			# check the current list of workers
			crashed_jobs = set()

			all_workers = list(self.worker_pool.keys())
			for wn in all_workers:
				# remove dead entries from the nameserver
				if not self.worker_pool[wn].is_alive():
					self.logger.info('DISPATCHER: removing dead worker, %s'%wn)
					update = True
					# todo check if there were jobs running on that that need to be rescheduled

					current_job = self.worker_pool[wn].runs_job

					if not current_job is None:
						self.logger.info('Job %s was not completed'%str(current_job))
						crashed_jobs.add(current_job)

					del self.worker_pool[wn]
					self.idle_workers.discard(wn)
					continue
					
				if not self.worker_pool[wn].is_busy():
					self.idle_workers.add(wn)


			# try to submit more jobs if something changed
			if update:
				if not self.queue_callback is None:
					self.discover_cond.release()
					self.queue_callback(len(self.worker_pool))
					self.discover_cond.acquire()
				self.runner_cond.notify()

			for crashed_job in crashed_jobs:
				self.discover_cond.release()
				self.register_result(crashed_job, {'result': None, 'exception': 'Worker died unexpectedly.'})
				self.discover_cond.acquire()

			self.logger.debug('DISPATCHER: Finished worker discovery')

			#if (len(self.worker_pool) == 0 ): # ping for new workers if no workers are currently available
			#	self.logger.debug('No workers available! Keep pinging')
			#	self.discover_cond.wait(sleep_interval)
			#	sleep_interval *= 2
			#else:
			self.discover_cond.wait(self.ping_interval)

			if self.shutdown_all_threads:
				self.logger.debug('DISPATCHER: discover_workers shutting down')
				self.runner_cond.notify()
				self.discover_cond.release()
				return

	def number_of_workers(self):
		with self.discover_cond:
			return(len(self.worker_pool))

	def job_runner(self):
		
		self.runner_cond.acquire()
		while True:
			
			while self.waiting_jobs.empty() or len(self.idle_workers) == 0:
				self.logger.debug('DISPATCHER: jobs to submit = %i, number of idle workers = %i -> waiting!'%(self.waiting_jobs.qsize(),  len(self.idle_workers) ))
				self.runner_cond.wait()
				self.logger.debug('DISPATCHER: Trying to submit another job.')
				if self.shutdown_all_threads:
					self.logger.debug('DISPATCHER: job_runner shutting down')
					self.discover_cond.notify()
					self.runner_cond.release()
					return
			
			job = self.waiting_jobs.get()
			wn = self.idle_workers.pop()

			worker = self.worker_pool[wn]
			self.logger.debug('DISPATCHER: starting job %s on %s'%(str(job.id),worker.name))
		
			job.time_it('started')
			worker.runs_job = job.id
		
			worker.proxy.start_computation(self, job.id, **job.kwargs)

			job.worker_name = wn
			self.running_jobs[job.id] = job

			self.logger.debug('DISPATCHER: job %s dispatched on %s'%(str(job.id),worker.name))


	def submit_job(self, id, **kwargs):
		self.logger.debug('DISPATCHER: trying to submit job %s'%str(id))
		with self.runner_cond:
			job = Job(id, **kwargs)
			job.time_it('submitted')
			self.waiting_jobs.put(job)
			self.logger.debug('DISPATCHER: trying to notify the job_runner thread.')
			self.runner_cond.notify()

	@Pyro4.expose
	@Pyro4.callback
	@Pyro4.oneway
	def register_result(self, id=None, result=None):
		self.logger.debug('DISPATCHER: job %s finished'%(str(id)))
		with self.runner_cond:
			self.logger.debug('DISPATCHER: register_result: lock acquired')
			# fill in missing information
			job = self.running_jobs[id]
			job.time_it('finished')
			job.result = result['result']
			job.exception = result['exception']

			self.logger.debug('DISPATCHER: job %s on %s finished'%(str(job.id),job.worker_name))
			self.logger.debug(str(job))
			
			# delete job
			del self.running_jobs[id]

			# label worker as idle again
			try:
				self.worker_pool[job.worker_name].runs_job = None
				self.worker_pool[job.worker_name].proxy._pyroRelease()
				self.idle_workers.add(job.worker_name)
				# notify the job_runner to check for more jobs to run
				self.runner_cond.notify()
			except KeyError:
				# happens for crashed workers, but we can just continue
				pass
			except:
				raise

		# call users callback function to register the result
		# needs to be with the condition released, as the master can call
		# submit_job quickly enough to cause a dead-lock
		self.new_result_callback(job)
