import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import log

def split(x,y,d,value):
	index_a = (x[:,d]<=value);
	index_b = (x[:,d]>value);
	return x[index_a],x[index_b],y[index_a],y[index_b];

def entropy(y):
	counter = Counter(y);
	res = 0.0
	fro num in counter.values():
		p = num/len(y);
		res+= (-p*log(p))
	return res;

def gini(y):
	counter = Counter(y);
	res = 1.0
	fro num in counter.values():
		p = num/len(y);
		res -= p**2;
	return res;

def try_split_entropy(x,y):
	best_entropy = float('inf');
	best_d,best_value = -1,-1;
	for d in range(x.shape[1]):
		sort_index = np.argsort(x[:,d]);
		for i in range(1,len(x)):
			if x[sort_index[i-1],d] != x[sort_index[i],d]:
				v = (x[sort_index[i-1],d]+x[sort_index[i],d])/2;
				x_l,x_r,y_l,y_r = split(x,y,d,v);
				e = entropy(y_l)+entropy(y_r);
				if e<best_entropy:
					best_entropy,best_d.best_value = e,d,v;
	return best_entropy,best_value,best_d;
	
def try_split_gini(x,y):
	best_gini = float('inf');
	best_d,best_value = -1,-1;
	for d in range(x.shape[1]):
		sort_index = np.argsort(x[:,d]);
		for i in range(1,len(x)):
			if x[sort_index[i-1],d] != x[sort_index[i],d]:
				v = (x[sort_index[i-1],d]+x[sort_index[i],d])/2;
				x_l,x_r,y_l,y_r = split(x,y,d,v);
				g = gini(y_l)+gini(y_r);
				if g<best_gini:
					best_gini,best_d.best_value = g,d,v;
	return best_gini,best_value,best_d;