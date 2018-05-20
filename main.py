import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666);
X = np.empty((100,2));
X[:,0] = np.random.uniform(0.0,100.,size = 100);
X[:,1] = 0.75*X[:,0]+3.0+np.random.normal(0.,10.,size=100);

def demean(X):
	return X-np.mean(X,axis=0);
	
def J(X,w):
	return np.sum((X.dot(w))**2)/len(X);


def dJ_math(X,w):
	return (X.T.dot(X.dot(w)))*2./len(X);
	
def dJ_debug(X,w,epsilon=0.00001):
	res = np.empty(len(w));
	for i in range(len(w)):
		w_1 = w.copy();
		w_1[i]+=epsilon;
		w_2 = w.copy();
		w_2[i] -= epsilon;
		res[i] = (J(X,w_1)-J(X,w_2))/(2*epsilon);
	return res;

def mode(w):
	return w/np.linalg.norm(w);
	
def gradientasect(X,initial_w,eta,n_iters = 1e4,epsilon = 1e-8):
	i_iters = 0;
	X_new = X;
	w = mode(initial_w);
	while i_iters<n_iters:
		gradient = dJ_math(X_new,w);
		last_w = w;
		w = w + eta*gradient;
		w = mode(w);
		if abs(J(X_new,last_w)-J(X_new,w))<epsilon:
			break;
		i_iters+=1;
	return w;

def gradientasect_debug(X,initial_w,eta,n_iters = 1e4,epsilon = 1e-8):
	i_iters = 0;
	X_new = X;
	w = mode(initial_w);
	while i_iters<n_iters:
		gradient = dJ_debug(X_new,w);
		last_w = w;
		w = w + eta*gradient;
		w = mode(w);
		if abs(J(X_new,last_w)-J(X_new,w))<epsilon:
			break;
		i_iters+=1;
	return w;

# x2 = x - x.dot(w).reshape(-1,1)*w;
def new_datasets(X,w):
#	x2 = np.empty(X.shape);
#	for i in range(len(X)):
#		x2[i] = X[i] - (X[i].dot(w))*w;
	return X - X.dot(w).reshape(-1,1)*w;

def fist_n_components(n,X,eta = 0.01,n_iters = 1e4,epsilon = 1e-8):
	x_pca = X.copy()
	x_pca = demean(x_pca);
	res = [];
	for i in range(n):
		initial_w = np.random.random(x_pca.shape[1]);
		w = gradientasect(x_pca,initial_w,eta,n_iters,epsilon);
		res.append(w);
		x_pca = new_datasets(x_pca,w);
	return res;
	
print(fist_n_components(2,X));