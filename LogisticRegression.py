import numpy as np
import metrics

class LogisticRegression(object):
	def __init__(self):
		self.coef_ = None;
		self.inter_ = None;
		self._theta = None;
	
	def _Sigmoid(self,x):
		return 1./(1.+np.exp(-x));
	
	def fit(self,X_train,y_train,eta = 0.01,n_iters=1e4):
		def J(theta,X_b,y):
			y_hat = self._Sigmoid(X_b.dot(theta));
			try:
				return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/len(y);
			except:
				return float("inf");
		def dJ(theta,X_b,y):
			return ((X_b.T).dot(self._Sigmoid(X_b.dot(theta))-y))/len(X_b);
			
		def gradientdescent(initial_theta,X_b,y,eta_1 = 0.01,epsilon = 1e-8,n_iters_1 = 1e4):
			i_iters = 0;
			theta = initial_theta;
			while i_iters<n_iters_1:
				gradient = dJ(theta,X_b,y);
				last_theta = theta;
				theta = theta - eta_1*gradient;
				if abs(J(theta,X_b,y)-J(last_theta,X_b,y))<epsilon:
					break;
			return theta;
		X_b = np.hstack([np.ones([len(X_train),1]),X_train]);
		theta = np.zeros([X_b.shape[1],1]);
		y_train = y_train.reshape(len(y_train),1);
		self._theta = gradientdescent(theta,X_b,y_train,eta_1=eta,n_iters_1=n_iters);
		self._theta = self._theta.reshape(len(self._theta));
		self.intercept_ = self._theta[0];
		self.coef_ = self._theta[1:];
		return self;
		
	def _predict_proba(self,X_test):
		X_b = np.hstack([np.ones([len(X_test),1]),X_test]);
		y_predict = self._Sigmoid(X_b.dot(self._theta));
		return y_predict;
	
	def predict(self,X_test):
		X_b = np.hstack([np.ones([len(X_test),1]),X_test]);
		proba = self._predict_proba(X_test);
		return np.array(proba>=0.5,dtype = 'int');

	def accuracy_score(self,x_test,y_test):
		_y_predict = self.predict(x_test);
		return metrics.accuracy_score(y_test,_y_predict);
		
	def precision_score(self,x_test,y_test):
		_y_predict = self.predict(x_test);
		return metrics.precision_score(y_test,_y_predict);
		
	def confusion_matrix(self,x_test,y_test):
		_y_predict = self.predict(x_test);
		return metrics.Confusion_Matrix(y_test,_y_predict);
		
	def recall_score(self,x_test,y_test):
		_y_predict = self.predict(x_test);
		return metrics.recall_score(y_test,_y_predict);
	
	def F1_score(self,x_test,y_test);
		_y_predict = self.predict(x_test);
		return metrics.F1_score(y_test,_y_predict);
	
	def __repr__(self):
		return "LinearRegression";
