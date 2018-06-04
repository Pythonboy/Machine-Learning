'''
作者： 楼浩然
功能： 将传入的样本随机分割为训练集和测试集
'''

import numpy as np

def train_test_split(X,Y,test_ratio = 0.2,random_state = None):
	assert X.shape[0] == Y.shape[0];      #保证样本数和结果数相等
	assert 0.0<= test_ratio <=1.0;        #保证分离训练集和测试集的有效性
	if random_state:                      #传入随机种子，可以使两次不同的训练、测试集的划分一致
		np.random.seed(random_state);
	shuffle_index = np.random.permutation(len(X));      #索引随机化过程
	test_size = int(len(X)*test_ratio);
	test_index = shuffle_index[:test_size];
	train_index = shuffle_index[test_size:];
	x_train = X[train_index];
	y_train = Y[train_index];
	x_test = X[test_index];
	y_test = Y[test_index];
	return x_train,x_test,y_train,y_test;   
	
