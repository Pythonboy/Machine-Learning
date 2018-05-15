import numpy as np

def train_test_split(X,Y,ratio = 0.2):
	assert X.shape[0] == Y.shape[0];
	assert 0.0<=ratio<=1.0;
	shuffle_index = np.random.permutation(len(X));
	test_size = int(len(X)*ratio);
	test_index = shuffle_index[:test_size];
	train_index = shuffle_index[test_size:];
	x_train = X[train_index];
	y_train = Y[train_index];
	x_test = X[test_index];
	y_test = Y[test_index];
	return x_train,y_train,x_test,y_test;
	
