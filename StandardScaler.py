'''
作者：楼浩然
功能：对数据集进行归一化处理 ————  最值归一化 & 均值方差归一化
'''

import numpy as np
#均值方差归一化
class StandardScaler(object):
	def __init__(self):         
		self._mean = None;
		self._scale = None;
		
	def fit(self,X):
		self._mean = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])     #求出数据集每列的均值
		self._scale = np.array([np.std(X[:,i]) for i in range(X.shape[1])])     #求出数据集每列的方差
		return self;
		
	def transform(self,X):
		x = np.empty(X.shape,dtype=float);
		for i in range(X.shape[1]):
			x[:,i] = 1.0*(X[:,i]-self._mean[i])/self._scale[i]         #归一化处理
		return x;
	
#最值归一化
class Normalization(object):
	def __init__(self):
		self._min = None;
		self._max = None;
	
	def fit(self,X):
		self._min = np.array([np.min(X[:,i] for i in range(X.shape[1]))]);     #求出数据集每列的最小值
		self._max = np.array([np.max(X[:,i] for i in range(X.shape[1]))]);     #求出数据集每列的最大值
		
	def transform(self,X):
		x = np.empty(X.shape,detype = float);
		for i in range(X.shape[1]):
			x[:,i] = 1.0 * (X[:,i] - self._min[i])/(self._max[i] - self._min[i]);     #归一化处理
		return x;
