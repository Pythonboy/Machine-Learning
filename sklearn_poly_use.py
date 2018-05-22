#-*- coding:utf-8 -*-
from sklearn.preprocessing import PolynomialFeatures  #多项式回归训练集转化
from sklearn.linear_model import LinearRegression    #线性回归
from sklearn.preprocessing import StandardScaler     #均值方差归一化
from sklearn.metrics import mean_squared_error       #均方误差
from sklearn.pipeline import Pipeline                 #管道输出
from sklearn.model_selection import train_test_split   #训练集和测试集分割
from sklearn.neighbors import KNeighborsClassifier    #KNN算法
from sklearn import datasets          #导入外部数据集
from sklearn.model_selection import cross_val_score    #交叉验证
from sklearn.linear_model import Ridge             #岭回归
from sklearn.linear_model import Lasso             #LASSO回归
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(666);
x = np.random.uniform(-3.,3.,size=(100,1));
y = 0.5*x**2+x+3+np.random.normal(0,1,size = (100,1));

#PolynomialRegression & Pipeline
def PolynomialRegression(degree):
	return Pipeline([
	("poly",PolynomialFeatures(degree=degree)),
	("std_scaler",StandardScaler()),
	("lin_reg",LinearRegression())
]);

#train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 10);
#LinearRegression
#underfitting
LR = LinearRegression();
LR.fit(x_train,y_train);
y_predict = LR.predict(x_test);
mse_lr = mean_squared_error(y_test,y_predict);
#PolynomialRegression,different degree
ploy_2 = PolynomialRegression(2);
ploy_2.fit(x_train,y_train);
y_predict_2 = ploy_2.predict(x_test);
mse_poly_2 = mean_squared_error(y_test,y_predict_2);
#overfitting
ploy_10 = PolynomialRegression(10);
ploy_10.fit(x_train,y_train);
y_predict_10 = ploy_10.predict(x_test);
mse_poly_10 = mean_squared_error(y_test,y_predict_10);
#print(mse_lr);
#print(mse_poly_2);
#print(mse_poly_10);


#learning curve
def plot_learning_curve(algo,x_train,x_test,y_train,y_test):
	train_score=[];
	test_score = [];
	for i in range(1,len(x_train)+1):
		algo.fit(x_train[:i],y_train[:i]);
		y_train_predict = algo.predict(x_train[:i]);
		y_test_predict = algo.predict(x_test);
		train_score.append(mean_squared_error(y_train[:i],y_train_predict[:i]));
		test_score.append(mean_squared_error(y_test,y_test_predict));
	plt.figure()
	plt.plot([i for i in range(1,len(x_train)+1)],np.sqrt(train_score),label = "train score");
	plt.plot([i for i in range(1,len(x_train)+1)],np.sqrt(test_score),label = "test score");
	plt.legend(loc="best");
	plt.axis([0,len(x_train)+1,0,4]);
	plt.show();
linearregression = LinearRegression();
#plot_learning_curve(linearregression,x_train,x_test,y_train,y_test);
poly_reg = PolynomialRegression(degree = 20);
#plot_learning_curve(poly_reg,x_train,x_test,y_train,y_test);


#PolynomialFeatures & LinearRegression
poly = PolynomialFeatures(degree=2);
poly.fit(x);
x_ = poly.transform(x);
linearregression = LinearRegression();
linearregression.fit(x_,y);
y_predict = linearregression.predict(x_);
score = linearregression.score(x_,y);

#print(score);
#print(linearregression.coef_);
#print(linearregression.intercept_);


#PolynomialRegression
poly_reg = PolynomialRegression(degree = 2);
poly_reg.fit(x,y);
y_predict_2 = poly_reg.predict(x);
mse_poly = mean_squared_error(y,y_predict_2);
#print(mse_poly);

#x=x.reshape(-1);
#plt.figure();
#plt.scatter(x,y);
#plt.plot(np.sort(x),y_predict[np.argsort(x)],'r',marker="+");
#plt.plot(np.sort(x),y_predict_2[np.argsort(x)],c ='b');
#plt.show();


#cross validation
digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4, random_state = 666);
best_score, best_k, best_p = 0,0,0
for k in range(2,11):
	for p in range(1,6):
		knn_clf = KNeighborsClassifier(weights = "distance",n_neighbors = k, p = p);
		scores = cross_val_score(knn_clf,x_train,y_train,cv=3);
		score = np.mean(scores);
		if score>best_score:
			best_score = score;
			best_k = k;
			best_p = p;
#print("best K:",best_k);
#print("best p:",best_p);
#print("best score:",best_score);
best_knn_clf = KNeighborsClassifier(weights = "distance",n_neighbors = best_k, p=best_p);
best_knn_clf.fit(x_train,y_train);
best_knn_clf.score(x_test,y_test);

#RidgeRegression
def RidgeRegression(degree,alpha):
	return Pipeline([
	("poly",PolynomialFeatures(degree=degree)),
	("std_scaler",StandardScaler()),
	("Ridge_reg",Ridge(alpha = alpha))
]);
ridge_reg_1 = RidgeRegression(20,0.001);
ridge_reg_1.fit(x_train,y_train);
y_predict = ridge_reg_1.predict(x_test);

#LASSO
def LassoRegression(degree,alpha):
	return Pipeline([
	("poly",PolynomialFeatures(degree=degree)),
	("std_scaler",StandardScaler()),
	("Lasso_reg",Lasso(alpha = alpha))
]);
lasso_reg_1 = LassoRegression(20,0.01);
lasso_reg_1.fit(x_train,y_train);
y_predict = lasso_reg_1.predict(x_test);
