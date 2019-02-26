import numpy as np
import scipy.optimize as spo

import ConfigSpace as CS

from hpbandster.optimizers.kde.kernels import Gaussian, AitchisonAitken, WangRyzinOrdinal, WangRyzinInteger


class MultivariateKDE(object):
	def __init__(self, configspace, fully_dimensional=True, min_bandwidth=1e-4, fix_boundary=True):
		"""
		Parameters:
		-----------
			configspace: ConfigSpace.ConfigurationSpace object
				description of the configuration space
			fully_dimensional: bool
				if True, a true multivariate KDE is build, otherwise it's approximated by
				the product of one dimensional KDEs
				
			min_bandwidth: float
				a lower limit to the bandwidths which can insure 'uncertainty'
		
		"""
		self.configspace = configspace
		self.types, self.num_values = self._get_types()
		self.min_bandwidth=min_bandwidth
		self.fully_dimensional=fully_dimensional
		self.fix_boundary = fix_boundary
		
		
		# precompute bandwidth bounds
		self.bw_bounds = []
		
		max_bw_cont=0.5
		max_bw_cat = 0.999
		
		for t in self.types:
			if t == 'C':
				self.bw_bounds.append((min_bandwidth, max_bw_cont))
			else:
				self.bw_bounds.append((min_bandwidth, max_bw_cat))
		
		self.bw_clip = np.array([ bwb[1] for bwb in self.bw_bounds ])

		# initialize other vars
		self.bandwidths = np.array([float('NaN')]*len(self.types))
		self.kernels = []
		for t,n in zip(self.types, self.num_values):
			
			kwargs = {'num_values':n, 'fix_boundary':fix_boundary}
			
			if t == 'I':	self.kernels.append(WangRyzinInteger(**kwargs))
			if t == 'C':	self.kernels.append(Gaussian(**kwargs))
			if t == 'O':	self.kernels.append(WangRyzinOrdinal(**kwargs))
			if t == 'U':	self.kernels.append(AitchisonAitken(**kwargs))			
		self.data = None
		
	
	
	def fit(self, data, weights=None, bw_estimator='scott', efficient_bw_estimation=True, update_bandwidth=True):
		"""
			fits the KDE to the data by estimating the bandwidths and storing the data
			
			Parameters
			----------
				data: 2d-array, shape N x M
					N datapoints in an M dimensional space to which the KDE is fit
				weights: 1d array
					N weights, one for every data point.
					They will be normalized to sum up to one
				fix_boundary_effects: bool
					whether to reweigh points close to the bondary no fix the pdf
				bw_estimator: str
					allowed values are 'scott' and 'mlcv' for Scott's rule of thumb
					and the maximum likelihood via cross-validation
				efficient_bw_estimation: bool
					if true, start bandwidth optimization from the previous value, otherwise
					start from Scott's values
				update_bandwidths: bool
					whether to update the bandwidths at all
		"""
		
		if self.data is None:
			# overwrite some values in case this is the first fit of the KDE
			efficient_bw_estimation = False
			update_bandwidth=True

		self.data = np.asfortranarray(data)
		for i,k in enumerate(self.kernels):
				self.kernels[i].data = self.data[:,i]

		self.weights = self._normalize_weights(weights)

		if not update_bandwidth:
			return
		
		if not efficient_bw_estimation or bw_estimator == 'scott':
			# inspired by the the statsmodels code
			sigmas = np.std(self.data, ddof=1, axis=0)
			IQRs = np.subtract.reduce(np.percentile(self.data, [75,25], axis=0))
			self.bandwidths = 1.059 * np.minimum(sigmas, IQRs) * np.power(self.data.shape[0], -0.2)
			# crop bandwidths for categorical parameters
			self.bandwidths = np.clip(self.bandwidths , self.min_bandwidth, self.bw_clip)
			
		if bw_estimator == 'mlcv':
			# optimize bandwidths here
			def opt_me(bw):
				self.bandwidths=bw
				self._set_kernel_bandwidths()
				return(self.loo_negloglikelihood())
			
			res = spo.minimize(opt_me, self.bandwidths, bounds=self.bw_bounds, method='SLSQP')
			self.optimizer_result = res
			self.bandwidths[:] = res.x
		self._set_kernel_bandwidths()

	def _set_kernel_bandwidths(self):
		for i,b in enumerate(self.bandwidths):
			self.kernels[i].set_bandwidth(b)


	def set_bandwidths(self, bandwidths):
		self.bandwidths[:] = bandwidths
		self._set_kernel_bandwidths()

	def _normalize_weights(self, weights):
		
		weights = np.ones(self.data.shape[0]) if weights is None else weights
		weights /= weights.sum()
		
		return(weights)


	def _individual_pdfs(self, x_test):
	
		pdfs = np.zeros(shape=[x_test.shape[0], self.data.shape[0], self.data.shape[1]], dtype=np.float)

		for i, k in enumerate(self.kernels):
			pdfs[:,:,i] = k(x_test[:,i]).T
		
		return(pdfs)


	def loo_negloglikelihood(self):
		# get all pdf values of the training data (including 'self interaction')
		pdfs = self._individual_pdfs(self.data)
		
		# get indices to remove diagonal values for LOO part :)
		indices = np.diag_indices(pdfs.shape[0])

		# combine values based on fully_dimensional!
		if self.fully_dimensional:
			pdfs[indices] = 0 # remove self interaction		
			
			pdfs2 = np.sum(np.prod(pdfs, axis=-1), axis=-1)
			sm_return_value = -np.log(pdfs2).sum()
			pdfs = np.prod(pdfs, axis=-1)
			
			# take weighted average (accounts for LOO!)
			lhs = np.sum(pdfs*self.weights, axis=-1)/(1-self.weights)
		else:
			#import pdb; pdb.set_trace()
			pdfs[indices] = 0 # we sum first so 0 is the appropriate value
			pdfs *= self.weights[:,None,None]
			
			pdfs = pdfs.sum(axis=-2)/(1-self.weights[:,None])
			lhs = np.prod(pdfs, axis=-1)
			
		return(-np.sum(self.weights*np.log(lhs)))


	def pdf(self, x_test):
		"""
			Computes the probability density function at all x_test
		"""
		N,D = self.data.shape
		x_test = np.asfortranarray(x_test)
		x_test = x_test.reshape([-1, D])
		
		pdfs = self._individual_pdfs(x_test)
		#import pdb; pdb.set_trace()
		# combine values based on fully_dimensional!
		if self.fully_dimensional:
			# first the product of the individual pdfs for each point in the data across dimensions and then the average (factorized kernel)
			pdfs = np.sum(np.prod(pdfs, axis=-1)*self.weights[None, :], axis=-1)
		else:
			# first the average over the 1d pdfs and the the product over dimensions (TPE like factorization of the pdf)
			pdfs = np.prod(np.sum(pdfs*self.weights[None,:,None], axis=-2), axis=-1)
		return(pdfs)


	def sample(self, num_samples=1):
		
		
		samples = np.zeros([num_samples, len(self.types)], dtype=np.float)

		if self.fully_dimensional:
			sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
			
		else:
			sample_indices=None

		for i,k in enumerate(self.kernels):
			samples[:,i] = k.sample(sample_indices, num_samples)


		return(samples)
		

	def _get_types(self):
		""" extracts the needed types from the configspace for faster retrival later
		
			type = 0 - numerical (continuous or integer) parameter
			type >=1 - categorical parameter
			
			TODO: figure out a way to properly handle ordinal parameters
		
		"""
		types = []
		num_values = []
		for hp in self.configspace.get_hyperparameters():
			#print(hp)
			if isinstance(hp, CS.CategoricalHyperparameter):
				types.append('U')
				num_values.append(len(hp.choices))
			elif isinstance(hp, CS.UniformIntegerHyperparameter):
				types.append('I')
				num_values.append((hp.upper - hp.lower + 1))
			elif isinstance(hp, CS.UniformFloatHyperparameter):
				types.append('C')
				num_values.append(np.inf)
			elif isinstance(hp, CS.OrdinalHyperparameter):
				types.append('O')
				num_values.append(len(hp.sequence))
			else:
				raise ValueError('Unsupported Parametertype %s'%type(hp))
		return(types, num_values)
