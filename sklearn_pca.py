#-*- coding:utf-8 -*-
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets;
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#digits数据集分类
digits = datasets.load_digits();
x = digits.data;
y = digits.target;
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 666);
KNN = KNeighborsClassifier();
KNN.fit(x_train,y_train);
sc0 = KNN.score(x_test,y_test);
print(sc0);

pca = PCA(0.90);
pca.fit(x_train);
x_train_reduction = pca.transform(x_train);
x_test_reduction = pca.transform(x_test);
#print(pca.explained_variance_ratio_);
print(pca.n_components_);

KNN1 = KNeighborsClassifier();
KNN1.fit(x_train_reduction,y_train);
sc1 = KNN1.score(x_test_reduction,y_test);
print(sc1);



















#自己创建数据集模拟
#np.random.seed(666);
#X = np.empty((100,2));
#X[:,0] = np.random.uniform(0.0,100.,size = 100);
#X[:,1] = 0.75*X[:,0]+3.0+np.random.normal(0.,10.,size=100);

#pca = PCA(n_components = 1);
#pca.fit(X);
#print(pca.components_);
#x_reduction = pca.transform(X);
#x_restore = pca.inverse_transform(x_reduction);
#plt.figure();
#plt.scatter(X[:,0],X[:,1],color="red",alpha = 0.5);
#plt.scatter(x_restore[:,0],x_restore[:,1],color="blue",alpha = 0.5);
#plt.show();