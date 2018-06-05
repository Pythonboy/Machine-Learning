'''
作者：楼浩然
功能：通过正规方程解和梯度下降法进行线性回归拟合
'''

import numpy as np
from metrics import r2_score

class LinearRegression(object):
	def __init__(self):
	'''初始化线性回归模型 '''
		self.coef_ = None;
		self.inter_ = None;
		self._theta = None;
		
	def fit_normal(self,X_train,y_train):
	'''通过正规方程解公式求出线性回归模型中相应的系数向量  '''
		assert len(X_train)==len(y_train);
		X_b = np.hstack([np.ones([len(X_train),1]),X_train]);
		self._theta = ((np.linalg.inv(X_b.T.dot(X_b))).dot(X_b.T)).dot(y_train);
		self.intercept_ = self._theta[0];
		self.coef_ = self._theta[1:];
		return self;
	
	def fit_gd(self,X_train,y_train,eta = 0.01,n_iters=1e5):
		def J(theta,X_b,y):
			try:
				return np.sum((X_b.dot(theta)-y)**2)/len(X_b);
			except:
				return float("inf");
			
		def dJ(theta,X_b,y):
			return ((X_b.T).dot(X_b.dot(theta)-y))*2/len(X_b);
			
		def gradientdescent(initial_theta,X_b,y,eta_1 = 0.01,epsilon = 1e-8,n_iters_1 = 1e5):
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
	
	def fit_sgd(self,X_train,y_train,n_iters=5):
		def learning_rate(i_iters):
			t0 = 5;
			t1 = 50;
			return t0/(i_iters+t1);
			
		def dJ_sgd(X_b_i,y_i,theta):
			return ((X_b_i.T).dot(X_b_i.dot(theta)-y_i))*2;

		def sgd(X_b,y,initial_theta,n_iters_1):
			theta = initial_theta;
			i_iters = 0;
			m = len(X_b);
			for i_iters in range(n_iters):
				index = np.random.permutation(m);
				X_b_new = X_b[index];
				y_new = y[index];
				for i in range(m):
					eta = learning_rate(i_iters*m+i);
					X_b_i = X_b_new[i].reshape(1,len(X_b_new[i]));
					y_i = y_new[i].reshape(1,len(y_new[i]));
					gradient = dJ_sgd(X_b_i,y_i,theta);
					theta = theta - eta*gradient;
			return theta;
		X_b = np.hstack([np.ones([len(X_train),1]),X_train]);
		theta = np.zeros([X_b.shape[1],1]);
		y_train = y_train.reshape(len(y_train),1);
		self._theta = sgd(X_b,y_train,theta,n_iters_1=n_iters);
		self._theta = self._theta.reshape(len(self._theta));
		self.intercept_ = self._theta[0];
		self.coef_ = self._theta[1:];
		return self;
		
	def predict(self,X_test):
	'''线性回归预测结果 '''
		X_b = np.hstack([np.ones([len(X_test),1]),X_test]);
		y_predict = X_b.dot(self._theta);
		return y_predict;
	
	def score(self,X_test,y_test):
		y_predict = self.predict(X_test);
		return r2_score(y_test,y_predict);
		
	def __repr__(self):
		return "LinearRegression";
