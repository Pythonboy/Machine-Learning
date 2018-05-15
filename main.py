#-*- conding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from KNeighborClassifier import KNNClassifier
from train_test_split import train_test_split
from StandardScaler import StandardScaler
#构建训练集和测试集
iris = datasets.load_iris();
x = iris.data
y = iris.target
X_train,Y_train,X_test,y_test = train_test_split(x,y);
standardscaler = StandardScaler();
standardscaler.fit(X_train);
x_train = standardscaler.transform(X_train);
x_test = standardscaler.transform(X_test);
KNN = KNNClassifier(k=4);
KNN.fit(x_train,Y_train);
y_predict = KNN.predict(x_test);
Score = KNN.score(x_test,y_test);
print(y_predict);
print(y_test);
print(Score);




