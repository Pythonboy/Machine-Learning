'''
作者： 楼浩然
功能： KNN算法的原理，可以对训练集进行分类，并判断分类的准确度
'''

import numpy as np
from collections import Counter
from math import sqrt

class KNNClassifier(object):
	#初始化训练集和K值
	def __init__(self,k):
		self.k = k;
		self._x_train = None;
		self._y_train = None;
	
	#训练集拟合过程
	def fit(self,X_train,Y_train):
		assert X_train.shape[0] == Y_train.shape[o];    #判断X_train训练集和Y_train结果的行数相同；
		assert k <= X_train.shape[0];    #判断K值要小于训练集的样本数
		self._x_train = X_train;         
		self._y_train = Y_train;
		return self;
	
	#测试集预测过程
	def predict(self,X_test):
		assert X_test.shape[1] == self._x_train.shape[1];   #判断预测集和训练集的特征数相同
		assert self._x_train is not None and self._y_train is not None;   #确保训练集已经进行过拟合
		y_predict = [self._predict(x_test) for x_test in X_test];    #根据测试集预测分类结果
		return y_predict;
	
	#根据单个待测数据，预测结果
	def _predict(self,x):
		distance = [sqrt(np.sum((x-x_train)**2)) for x_train in self._x_train];     #计算测试集中单个数据距离训练集各个样本的距离
		res = np.argsort(distance);                       #对距离进行从小到大的排序，并返回下标值
		y_like = [self._y_train[i] for i in res[:self.k]];       #取出距离列表中最小的K个值
		Res = Counter(y_like);              
		return Res.most_common(1)[0][0];      #返回K个值中数量最多的分类结果
	
	#算法的准确度
	def score(self,x_test,y_test):
		_y_predict = self.predict(x_test);
		sco = 1.0*sum(_y_predict==y_test)/len(y_test);         #预测正确的结果数量/测试集结果数量
		return sco;
		
	def __repr__(self):
		return "KNN(k=%d)"%self.k;
