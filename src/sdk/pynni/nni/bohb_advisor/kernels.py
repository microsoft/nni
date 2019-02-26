import numpy as np
import scipy.optimize as spo

import ConfigSpace as CS

from scipy.special import erf

from pdb import set_trace


class BaseKernel(object):
    def __init__(self, data=None, bandwidth=None, fix_boundary=False, num_values=None):
	
    	self.data = data
    	self.bw = bandwidth
    	self.fix_boundary=fix_boundary
    
    	if num_values is None:
            num_values = len(np.unique(data))
        self.num_values = num_values
		
		if not data is None:
			self.weights = self._compute_weights()	

	def set_bandwidth(self, bandwidth):
		self.bw = bandwidth
		self.weights = self._compute_weights()

	def _compute_weights(self):
		return(1.)

	def __call__(self, x_test):
		raise NotImplementedError

	def sample(self, sample_indices=None, num_samples=1):
		raise NotImplementedError


class Gaussian(BaseKernel):

	def _compute_weights(self):
		if not self.fix_boundary:
			return(1.)

		weights = np.zeros(self.data.shape[0])
		for i,d in enumerate(self.data):
			weights[i] = 2./(erf((1-d)/(np.sqrt(2)*self.bw)) + erf(d/(np.sqrt(2)*self.bw)))
		
		return(weights[:,None])
		
	def __call__(self, x_test):
		distances = x_test[None,:] - self.data[:,None]
		pdfs = np.exp(-0.5* np.power(distances/self.bw, 2))/(2.5066282746310002 * self.bw)
		
		# reweigh to compensate for boundaries
		pdfs *= self.weights
		
		return(pdfs)

	def sample(self, sample_indices=None, num_samples=1):
		""" returns samples according to the KDE
		
			Parameters
			----------
				sample_inices: list of ints
					Indices into the training data used as centers for the samples
				
				num_samples: int
					if samples_indices is None, this specifies how many samples
					are drawn.
				
		"""
		if sample_indices is None:
			sample_indices = np.random.choice(self.data.shape[0], size=num_samples)

		samples = self.data[sample_indices]

		delta = np.random.normal(size=num_samples)*self.bw
		samples += delta
		oob_idx  = np.argwhere(np.logical_or(samples > 1, samples < 0)).flatten()

		while len(oob_idx) > 0:
			samples[oob_idx] -= delta[oob_idx]		# revert move
			delta[oob_idx] = np.random.normal(size=len(oob_idx))*self.bw
			samples[oob_idx] += delta[oob_idx]
			oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx] > 1, samples[oob_idx] < 0))].flatten()

		return(samples)

class AitchisonAitken(BaseKernel):
	def __call__(self, x_test):
		distances = np.rint(x_test[None,:] - self.data[:,None])

		idx = np.abs(distances) == 0
		distances[idx] = 1 - self.bw
		distances[~idx] = self.bw/(self.num_values-1)
		
		return(distances)


	def sample(self, sample_indices=None, num_samples=1):
		""" returns samples according to the KDE
		
			Parameters
			----------
				sample_inices: list of ints
					Indices into the training data used as centers for the samples
				
				num_samples: int
					if samples_indices is None, this specifies how many samples
					are drawn.
				
		"""
		if sample_indices is None:
			sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
		samples = self.data[sample_indices]

		samples = samples.squeeze()
		
		if self.num_values == 1:
			# handle cases where there is only one value!
			return(samples)
		
		probs = self.bw * np.ones(self.num_values)/(self.num_values-1)
		probs[0] = 1-self.bw
		
		delta = np.random.choice(self.num_values, size=num_samples, p = probs)
		samples = np.mod(samples + delta, self.num_values)	

		return(samples)

class WangRyzinOrdinal(BaseKernel):
		
	def _compute_weights(self):
		if not self.fix_boundary:
			return(1.)
		np.zeros(self.data.shape[0])
		self.weights=1.
		x_test = np.arange(self.num_values)
		pdfs  = self.__call__(x_test)
		weights = 1./pdfs.sum(axis=1)[:,None]
		return(weights)
		
	def __call__(self, x_test):
		distances = x_test[None,:] - self.data[:,None]

		idx = np.abs(distances) < .1 # distances smaller than that are considered zero
		
		pdfs = np.zeros_like(distances, dtype=np.float)
		pdfs[idx] = (1-self.bw)
		pdfs[~idx] = 0.5*(1-self.bw) * np.power(self.bw, np.abs(distances[~idx]))
		# reweigh to compensate for boundaries
		pdfs *= self.weights

		return(pdfs)

	def sample(self, sample_indices=None, num_samples=1):
		""" returns samples according to the KDE
		
			Parameters
			----------
				sample_inices: list of ints
					Indices into the training data used as centers for the samples
				
				num_samples: int
					if samples_indices is None, this specifies how many samples
					are drawn.
				
		"""
		if sample_indices is None:
			sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
			samples = self.data[sample_indices]

		possible_steps = np.arange(-self.num_values+1,self.num_values)
		idx = (np.abs(possible_steps) < 1e-2)
		
		ps = 0.5*(1-self.bw) * np.power(self.bw, np.abs(possible_steps))
		ps[idx] = (1-self.bw)
		ps /= ps.sum()
		
		delta = np.zeros_like(samples)
		oob_idx = np.arange(samples.shape[0])

		while len(oob_idx) > 0:
			samples[oob_idx] -= delta[oob_idx]		# revert move
			delta[oob_idx] = np.random.choice(possible_steps, size=len(oob_idx), p=ps)
			samples[oob_idx] += delta[oob_idx]
			#import pdb; pdb.set_trace()
			oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx] > self.num_values-0.9, samples[oob_idx] < -0.1)).flatten()]
		return(np.rint(samples))

class WangRyzinInteger(BaseKernel):
	def _compute_weights(self):
		if not self.fix_boundary:
			return(1.)
		weights = np.zeros(self.data.shape[0], dtype=np.float)
		x_test = np.linspace(1/(2*self.num_values), 1-(1/(2*self.num_values)), self.num_values, endpoint=True)
		self.weights = 1.
		pdfs  = self.__call__(x_test)
		weights = 1./pdfs.sum(axis=1)[:,None]
		return(weights)
		
	def __call__(self, x_test):
		distances = (x_test[None,:] - self.data[:,None])
		
		pdfs = np.zeros_like(distances, dtype=np.float)

		idx = np.abs(distances) < 1/(3*self.num_values) # distances smaller than that are considered zero
		pdfs[idx] = (1-self.bw)
		pdfs[~idx] = 0.5*(1-self.bw) * np.power(self.bw, np.abs(distances[~idx])*self.num_values)
		# reweigh to compensate for boundaries
		pdfs *= self.weights
				
		return(pdfs)

	def sample(self, sample_indices=None, num_samples=1):
		""" returns samples according to the KDE
		
			Parameters
			----------
				sample_inices: list of ints
					Indices into the training data used as centers for the samples
				
				num_samples: int
					if samples_indices is None, this specifies how many samples
					are drawn.
				
		"""
		if sample_indices is None:
			sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
		samples = self.data[sample_indices]

		possible_steps = np.arange(-self.num_values+1,self.num_values) / self.num_values
		ps = 0.5*(1-self.bw) * np.power(self.bw, np.abs(possible_steps))
		ps[self.num_values-1] = (1-self.bw)
		ps /= ps.sum()
		
		delta = np.zeros_like(samples)
		oob_idx = np.arange(samples.shape[0])

		while len(oob_idx) > 0:
			samples[oob_idx] -= delta[oob_idx]		# revert move
			delta[oob_idx] = np.random.choice(possible_steps, size=len(oob_idx), p=ps)
			samples[oob_idx] += delta[oob_idx]
			oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx] > 1-1/(3*self.num_values), samples[oob_idx] < 1/(3*self.num_values))).flatten()]

		
		return(samples)
