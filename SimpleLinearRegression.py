'''
作者： 楼浩然
功能：使用最小二乘法进行简单线性回归（y = a*x+b)
'''

import numpy as np
from metrics import r2_score

class SimpleLinearRegression(object):
	'''初始化简单线性回归模型 '''
	def __init__(self):
		self.a_ = None;
		self.b_ = None;
	
	#拟合过程
	def fit(self,x_train,y_train):
		assert x_train.shape == y_train.shape;
		x_mean = np.mean(x_train);
		y_mean = np.mean(y_train);
		self.a_ = (x_train-x_mean).dot(y_train-y_mean)/(x_train-x_mean).dot(x_train - x_mean);      #根据最小二乘法求出相应的系数a,b ； 向量化运算
		self.b_ = y_mean - self.a_*x_mean;
		return self;
	
	#预测过程
	def predict(self,x_test):
		assert x_test.ndim==1;
		y_predict = [self._predict(single) for single in x_test];
		return np.array(y_predict);
	
	#单个样本的回归预测过程
	def _predict(self,x):
		y = self.a_*x+self.b_;
		return y;
	
	#评价函数 ： R squared 评价函数
	def score(self,x_test,y_test):
		y_predict = self.predict(x_test);
		return r2_score(y_test,y_predict);
		
	def __repr__(self):
		return "SimpleLinearRegression";
