import numpy as np

def train_test_split(X,Y,ratio = 0.2):
	assert X.shape[0]==Y.shape[0];
	assert 0<=ratio<=1;
	shuffle_indexes = np.random.permutation(len(X));
	train_size = int(len(X)*(1-ratio));
	train_index = shuffle_indexes[:train_size];
	test_index = shuffle_indexes[train_size:];
	X_train = X[train_index];
	X_test = X[test_index];
	Y_train = Y[train_index];
	Y_test = Y[test_index];
	return X_train,X_test,Y_train,Y_test;
	