import numpy as np
from metrics import r2_score

class SimpleLinearRegression(object):
	def __init__(self):
		self.a_ = None;
		self.b_ = None;
		
	def fit(self,x_train,y_train):
		assert x_train.shape == y_train.shape;
		x_mean = np.mean(x_train);
		y_mean = np.mean(y_train);
		self.a_ = (x_train-x_mean).dot(y_train-y_mean)/np.sum((x_train-x_mean)**2)
		self.b_ = y_mean - self.a_*x_mean;
		return self;
		
	def predict(self,x_test):
		assert x_test.ndim==1;
		y_predict = [self._predict(single) for single in x_test];
		return np.array(y_predict);
		
	def _predict(self,x):
		y = self.a_*x+self.b_;
		return y;
		
	def score(self,x_test,y_test):
		y_predict = self.predict(x_test);
		return r2_score(y_test,y_predict);
		
	def __repr__(self):
		return "SimpleLinearRegression";