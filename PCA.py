import numpy as np
import numpy as np

class PCA(object):
	def __init__(self,n_components):
		self.n_components = n_components;
		self.components_ = None;
	
	def gradientascent(self,X,initial_w,eta=0.01,n_iters = 1e4,epsilon = 1e-8):

		def J(X,w):
			return np.sum((X.dot(w))**2)/len(X);
		def dJ(X,w):
			return X.T.dot(X.dot(w))*2./len(X);
		def direction(w):
			return w/np.linalg.norm(w);
		w = direction(initial_w);
		i_iters = 0;
		while i_iters<n_iters:
			gradient = dJ(X,w);
			last_w = w;
			w = w+eta*gradient;
			w = direction(w);
			if abs(J(X,last_w)-J(X,w))<epsilon:
				break;
			i_iters+=1;
		return w;
		
	def fit(self,X_origin,eta=0.01,n_iters=1e4,epsilon = 1e-8):
		self.components_ = np.empty((self.n_components,X_origin.shape[1]));
		def new_datasets(X,w):
			x2 = X-X.dot(w).reshape(-1,1)*w;
			return x2;
		def demean(X):
			return X-np.mean(X,axis = 0);
		x_pca = X_origin.copy();
		x_pca = demean(x_pca);
		for i in range(self.n_components):
			w = np.random.random(x_pca.shape[1]);
			w = self.gradientascent(x_pca,w,eta,n_iters,epsilon);
			self.components_[i,:] = w;
			x_pca = new_datasets(x_pca,w);
		return self;

	def transform(self,X):
		return X.dot(self.components_.T);
		
	def inverse_transform(self,X):
		return X.dot(self.components_);
	
	def __repr__(self):
		print("PCA")

				