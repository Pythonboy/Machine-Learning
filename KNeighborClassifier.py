import numpy as np
from collections import Counter
from math import sqrt

class KNNClassifier(object):
	def __init__(self,k):
		self.k = k;
		self._x_train = None;
		self._y_train = None;
		
	def fit(self,X_train,Y_train):
		self._x_train = X_train;
		self._y_train = Y_train;
		return self;
		
	def predict(self,X_test):
		y_predict = [self._predict(x_test) for x_test in X_test];
		return y_predict;
		
	def _predict(self,x):
		distance = [sqrt(np.sum((x-x_train)**2)) for x_train in self._x_train];
		res = np.argsort(distance);
		y_like = [self._y_train[i] for i in res[:self.k]];
		Res = Counter(y_like);
		return Res.most_common(1)[0][0];
		
	def score(self,x_test,y_test):
		_y_predict = self.predict(x_test);
		sco = 1.0*sum(_y_predict==y_test)/len(y_test);
		return sco;
		
	def __repr__(self):
		return "KNN(k=%d)"%self.k;