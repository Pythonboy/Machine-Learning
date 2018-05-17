from SimpleLinearRegression import SimpleLinearRegression
from train_test_split import train_test_split
import metrics as mt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as LR

#多元线性回归
boston = datasets.load_boston();
x = boston.data
y = boston.target
x = x[y<50];
y = y[y<50];
X_train,X_test,y_train,y_test = train_test_split(x,y,0.3);
lr = LR();
lr.fit(X_train,y_train);
co1 = lr.coef_
linearregression = LinearRegression();
linearregression.fit_normal(X_train,y_train);
y_predict = linearregression.predict(X_test);
score = linearregression.score(X_test,y_test);
MSE = mt.Mean_Squared_Error(y_test,y_predict);
mse = mean_squared_error(y_test,y_predict);
co2 = linearregression.coef_;
print("MSE:%f"%MSE);
print("mean_squared_error:%f"%mse);
print("R_Squared:%f"%score);
print("co1:");
print(co1);
print("co2:");
print(co2);

#简单线性回归
#boston = datasets.load_boston();
#x = boston.data[:,5];
#y = boston.target;
#x = x[y<50];
#y = y[y<50];
#x_train,x_test,y_train,y_test = train_test_split(x,y,0.3);
#simplelinearregression = SimpleLinearRegression();
#simplelinearregression.fit(x_train,y_train);
#y_predict = simplelinearregression.predict(x_test);
#MSE = mt.Mean_Squared_Error(y_test,y_predict);
#RMSE = mt.Root_Mean_Squared_Error(y_test,y_predict)
#MAE = mt.Mean_Absolute_Error(y_test,y_predict);
#R_2 = mt.r2_score(y_test,y_predict);
#r_2 = simplelinearregression.score(x_test,y_test);
#mse = mean_squared_error(y_test,y_predict);
#print("MSE:%f ; RMSE:%f ; MAE:%f ; R_2:%f ; r_2:%f"%(MSE,RMSE,MAE,R_2,r_2));
#print("mean_squared_error:%f"%mse);
#plt.figure();
#plt.scatter(x,y);
#plt.plot(x,simplelinearregression.predict(x),'r');
#plt.show();