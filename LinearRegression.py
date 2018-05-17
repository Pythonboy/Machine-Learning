import numpy as np
from metrics import r2_score

class LinearRegression(object):
	def __init__(self):
		self.coef_ = None;
		self.inter_ = None;
		self._theta = None;
		
	def fit_normal(self,X_train,y_train):
		assert len(X_train)==len(y_train);
		X_b = np.hstack([np.ones([len(X_train),1]),X_train]);
		self._theta = ((np.linalg.inv(X_b.T.dot(X_b))).dot(X_b.T)).dot(y_train);
		self.inter_ = self._theta[0];
		self.coef_ = self._theta[1:];
		return self;
		
	def predict(self,X_test):
		X_b = np.hstack([np.ones([len(X_test),1]),X_test]);
		y_predict = X_b.dot(self._theta);
		return y_predict;
	
	def score(self,X_test,y_test):
		y_predict = self.predict(X_test);
		return r2_score(y_test,y_predict);
		
	def __repr__(self):
		return "LinearRegression";