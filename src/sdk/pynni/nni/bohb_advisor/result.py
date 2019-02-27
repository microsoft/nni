import copy
import os
import json

from .base_iteration import  Datum

class Run(object):
    """
		Not a proper class, more a 'struct' to bundle important
		information about a particular run
	"""
    def __init__(self, config_id, budget, loss, info, time_stamps, error_logs):
    	self.config_id   = config_id
    	self.budget      = budget
    	self.error_logs  = error_logs
    	self.loss        = loss
    	self.info        = info
    	self.time_stamps = time_stamps

    def __repr__(self):
    	return(\
			"config_id: %s\t"%(self.config_id,) + \
			"budget: %f\t"%self.budget + \
			"loss: %s\n"%self.loss + \
			"time_stamps: {submitted} (submitted), {started} (started), {finished} (finished)\n".format(**self.time_stamps) + \
			"info: %s\n"%self.info
    	)
    def __getitem__ (self, k):
    	"""
			 in case somebody wants to use it like a dictionary
		"""
    	return(getattr(self, k))


def extract_HBS_learning_curves(runs):
    """
	function to get the hyperband learning curves

	This is an example function showing the interface to use the
	HB_result.get_learning_curves method.

	Parameters
	----------

	runs: list of HB_result.run objects
		the performed runs for an unspecified config

	Returns
	-------

	list of learning curves: list of lists of tuples
		An individual learning curve is a list of (t, x_t) tuples.
		This function must return a list of these. One could think
		of cases where one could extract multiple learning curves
		from these runs, e.g. if each run is an independent training
		run of a neural network on the data.
    
	"""
    sr = sorted(runs, key=lambda r: r.budget)
    lc = list(filter(lambda t: not t[1] is None, [(r.budget, r.loss) for r in sr]))
    return([lc,])





def logged_results_to_HBS_result(directory):
    """
	function to import logged 'live-results' and return a HB_result object

	You can load live run results with this function and the returned
	HB_result object gives you access to the results the same way
	a finished run would.

	Parameters
	----------
	directory: str
		the directory containing the results.json and config.json files

	Returns
	-------
	hpbandster.core.result.Result: :object:
		TODO

	"""
    data = {}
    time_ref = float('inf')
    budget_set = set()

    with open(os.path.join(directory, 'configs.json')) as fh:
    	for line in fh:
            
    		line = json.loads(line)
    
    		if len(line) == 3:
    			config_id, config, config_info = line
    		if len(line) == 2:
    			config_id, config, = line
    			config_info = 'N/A'

    		data[tuple(config_id)] = Datum(config=config, config_info=config_info)

    with open(os.path.join(directory, 'results.json')) as fh:
    	for line in fh:
    		config_id, budget,time_stamps, result, exception = json.loads(line)

    		id = tuple(config_id)
    
    		data[id].time_stamps[budget] = time_stamps
    		data[id].results[budget] = result
    		data[id].exceptions[budget] = exception

    		budget_set.add(budget)
    		time_ref = min(time_ref, time_stamps['submitted'])


	# infer the hyperband configuration from the data
    budget_list = sorted(list(budget_set))

    HB_config = {
    					'eta'        : None if len(budget_list) < 2 else budget_list[1]/budget_list[0],
    					'min_budget' : min(budget_set),
    					'max_budget' : max(budget_set),
    					'budgets'    : budget_list,
    					'max_SH_iter': len(budget_set),
    					'time_ref'   : time_ref
    			}
    return(Result([data], HB_config))


class Result(object):
    """
	Object returned by the HB_master.run function

	This class offers a simple API to access the information from
	a Hyperband run.
	"""
    def __init__ (self, HB_iteration_data, HB_config):
		self.data = HB_iteration_data
		self.HB_config = HB_config
		self._merge_results()

	def __getitem__(self, k):
		return(self.data[k])


	def get_incumbent_id(self):
		"""
		Find the config_id of the incumbent.

		The incumbent here is the configuration with the smallest loss
		among all runs on the maximum budget! If no run finishes on the
		maximum budget, None is returned!
		"""
		tmp_list = []
		for k,v in self.data.items():
			try:
				# only things run for the max budget are considered
				res = v.results[self.HB_config['max_budget']]
				if not res is None:
					tmp_list.append((res['loss'], k))
			except KeyError as e:
				pass
			except:
				raise

		if len(tmp_list) > 0:
			return(min(tmp_list)[1])
		return(None)



	def get_incumbent_trajectory(self, all_budgets=True, bigger_is_better=True, non_decreasing_budget=True):
		"""
		Returns the best configurations over time
		
		
		Parameters
		----------
			all_budgets: bool
				If set to true all runs (even those not with the largest budget) can be the incumbent.
				Otherwise, only full budget runs are considered
			bigger_is_better:bool
				flag whether an evaluation on a larger budget is always considered better.
				If True, the incumbent might increase for the first evaluations on a bigger budget
			non_decreasing_budget: bool
				flag whether the budget of a new incumbent should be at least as big as the one for
				the current incumbent.
		Returns
		-------
			dict:
				dictionary with all the config IDs, the times the runs
				finished, their respective budgets, and corresponding losses
		"""
		all_runs = self.get_all_runs(only_largest_budget = not all_budgets)
		
		if not all_budgets:
			all_runs = list(filter(lambda r: r.budget==res.HB_config['max_budget'], all_runs))
		
		all_runs.sort(key=lambda r: r.time_stamps['finished'])
		
		return_dict = { 'config_ids' : [],
						'times_finished': [],
						'budgets'    : [],
						'losses'     : [],
		}
	
		current_incumbent = float('inf')
		incumbent_budget = self.HB_config['min_budget']
		
		for r in all_runs:
			if r.loss is None: continue
			
			new_incumbent = False
			
			if bigger_is_better and r.budget > incumbent_budget:
				new_incumbent = True
			
			
			if r.loss < current_incumbent:
				new_incumbent = True
			
			if non_decreasing_budget and r.budget < incumbent_budget:
				new_incumbent = False
			
			if new_incumbent:
				current_incumbent = r.loss
				incumbent_budget  = r.budget
				
				return_dict['config_ids'].append(r.config_id)
				return_dict['times_finished'].append(r.time_stamps['finished'])
				return_dict['budgets'].append(r.budget)
				return_dict['losses'].append(r.loss)

		if current_incumbent != r.loss:
			r = all_runs[-1]
		
			return_dict['config_ids'].append(return_dict['config_ids'][-1])
			return_dict['times_finished'].append(r.time_stamps['finished'])
			return_dict['budgets'].append(return_dict['budgets'][-1])
			return_dict['losses'].append(return_dict['losses'][-1])

			
		return (return_dict)


	def get_runs_by_id(self, config_id):
		"""
		returns a list of runs for a given config id

		The runs are sorted by ascending budget, so '-1' will give
		the longest run for this config.
		"""
		d = self.data[config_id]

		runs = []
		for b in d.results.keys():
			try:
				err_logs = d.exceptions.get(b, None)

				if d.results[b] is None:
					r = Run(config_id, b, None, None , d.time_stamps[b], err_logs)
				else:
					r = Run(config_id, b, d.results[b]['loss'], d.results[b]['info'] , d.time_stamps[b], err_logs)
				runs.append(r)
			except:
				raise
		runs.sort(key=lambda r: r.budget)
		return(runs)


	def get_learning_curves(self, lc_extractor=extract_HBS_learning_curves, config_ids=None):
		"""
		extracts all learning curves from all run configurations

		Parameters
		----------
			lc_extractor: callable
				a function to return a list of learning_curves.
				defaults to hpbanster.HB_result.extract_HP_learning_curves
			config_ids: list of valid config ids
				if only a subset of the config ids is wanted

		Returns
		-------
			dict
				a dictionary with the config_ids as keys and the
				learning curves as values
		"""

		config_ids = self.data.keys() if config_ids is None else config_ids
		
		lc_dict = {}
		
		for id in config_ids:
			runs = self.get_runs_by_id(id)
			lc_dict[id] = lc_extractor(runs)
			
		return(lc_dict)


	def get_all_runs(self, only_largest_budget=False):
		"""
		returns all runs performed

		Parameters
		----------
			only_largest_budget: boolean
				if True, only the largest budget for each configuration
				is returned. This makes sense if the runs are continued
				across budgets and the info field contains the information
				you care about. If False, all runs of a configuration
				are returned
		"""
		all_runs = []

		for k in self.data.keys():
			runs = self.get_runs_by_id(k)

			if len(runs) > 0:
				if only_largest_budget:
					all_runs.append(runs[-1])
				else:
					all_runs.extend(runs)

		return(all_runs)

	def get_id2config_mapping(self):
		"""
		returns a dict where the keys are the config_ids and the values
		are the actual configurations
		"""
		new_dict = {}
		for k, v in self.data.items():
			new_dict[k] = {}
			new_dict[k]['config'] = copy.deepcopy(v.config)
			try:
				new_dict[k]['config_info'] = copy.deepcopy(v.config_info)
			except:
				pass
		return(new_dict)

	def _merge_results(self):
		"""
		hidden function to merge the list of results into one
		dictionary and 'normalize' the time stamps
		"""
		new_dict = {}
		for it in self.data:
			new_dict.update(it)

		for k,v in new_dict.items():
			for kk, vv in v.time_stamps.items():
				for kkk,vvv in vv.items():
					new_dict[k].time_stamps[kk][kkk] = vvv - self.HB_config['time_ref']

		self.data = new_dict

	def num_iterations(self):
		return(max([k[0] for k in self.data.keys()]) + 1)
		

	def get_fANOVA_data(self, config_space, budgets=None, loss_fn=lambda r: r.loss, failed_loss=None):

		import numpy as np
		import ConfigSpace as CS

		id2conf = self.get_id2config_mapping()

		if budgets is None:
			budgets = self.HB_config['budgets']

		if len(budgets)>1:
			config_space.add_hyperparameter(CS.UniformFloatHyperparameter('budget', min(budgets), max(budgets), log=True))
		
		hp_names = config_space.get_hyperparameter_names()
		hps = config_space.get_hyperparameters()
		needs_transform = list(map(lambda h: isinstance(h, CS.CategoricalHyperparameter), hps))

		all_runs = self.get_all_runs(only_largest_budget=False)


		all_runs=list(filter( lambda r: r.budget in budgets, all_runs))

		X = []
		y = []

		for r in all_runs:
			if r.loss is None:
				if failed_loss is None: continue
				else: y.append(failed_loss)
			else:
				y.append(loss_fn(r))
				
			config = id2conf[r.config_id]['config']
			if len(budgets)>1:
				config['budget'] = r.budget

			config = CS.Configuration(config_space, config)
			
			x = []
			for (name, hp, transform) in zip(hp_names, hps, needs_transform):
				if transform:
					x.append(hp._inverse_transform(config[name]))
				else:
					x.append(config[name])
			
			X.append(x)

		return(np.array(X), np.array(y), config_space)


	def get_pandas_dataframe(self, budgets=None, loss_fn=lambda r: r.loss):

		import numpy as np
		import pandas as pd

		id2conf = self.get_id2config_mapping()

		df_x = pd.DataFrame()
		df_y = pd.DataFrame()


		if budgets is None:
			budgets = self.HB_config['budgets']

		all_runs = self.get_all_runs(only_largest_budget=False)
		all_runs=list(filter( lambda r: r.budget in budgets, all_runs))



		all_configs = []
		all_losses = []

		for r in all_runs:
			if r.loss is None: continue
			config = id2conf[r.config_id]['config']
			if len(budgets)>1:
				config['budget'] = r.budget

			all_configs.append(config)
			all_losses.append({'loss': r.loss})
			
			#df_x = df_x.append(config, ignore_index=True)
			#df_y = df_y.append({'loss': r.loss}, ignore_index=True)
		
		df_X = pd.DataFrame(all_configs)
		df_y = pd.DataFrame(all_losses)

		return(df_X, df_y)

