import numpy as np

class StandardScaler(object):
	def __init__(self):
		self._mean = None;
		self._scale = None;
		
	def fit(self,X):
		self._mean = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
		self._scale = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
		return self;
		
	def transform(self,X):
		x = np.empty(X.shape,dtype=float);
		for i in range(X.shape[1]):
#			print(self._scale[i]);
			x[:,i] = 1.0*(X[:,i]-self._mean[i])/self._scale[i]
		return x;
			