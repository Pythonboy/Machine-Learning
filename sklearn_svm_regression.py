import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

#训练集&测试集
boston = datasets.load_boston();
x = boston.data 
y = boston.target
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =666);

#SVM中的线性回归
def StandardLinearSVR(C=10.0,epsilon=0.01):
	return Pipeline([
		("std_scaler",StandardScaler()),
		("linearSVR",LinearSVR(C = C,epsilon = epsilon))
		])
def StandardLinearRegression():
	return Pipeline([
		("std_scaler",StandardScaler()),
		("linearSVR",LinearRegression())
		])
svr = StandardLinearSVR();
svr.fit(x_train,y_train);
print(svr.score(x_test,y_test));

lr = StandardLinearRegression();
lr.fit(x_train,y_train);
print(lr.score(x_test,y_test));

# python sklearn_svm_regression.py