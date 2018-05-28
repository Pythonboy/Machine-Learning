#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#导入燕尾花数据集
iris = datasets.load_iris();
x = iris.data
y = iris.target
x = x[y<2,:2];
y = y[y<2];

#数据归一化处理
std = StandardScaler();
std.fit(x);
x_strandard = std.transform(x);

#SVM分析
svc = LinearSVC(C=1e9);
svc.fit(x_strandard,y);
print(svc.coef_);
print(svc.intercept_);

#决策边界绘制
def plot_decision_boundary(model,axis):
	x0,x1 = np.meshgrid(
		np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
		np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
	)
	x_new = np.c_[x0.ravel(),x1.ravel()];
	y_predict = model.predict(x_new);
	zz = y_predict.reshape(x0.shape);
	from matplotlib.colors import ListedColormap
	custom_cmap = ListedColormap(["#EF9A9A","#FFF59D","#90CAF9"])
	plt.contourf(x0,x1,zz,linewidth = 5,cmap = custom_cmap);

def plot_svc_decision_boundary(model,axis):
	x0,x1 = np.meshgrid(
		np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
		np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
	)
	x_new = np.c_[x0.ravel(),x1.ravel()];
	y_predict = model.predict(x_new);
	zz = y_predict.reshape(x0.shape);
	from matplotlib.colors import ListedColormap
	custom_cmap = ListedColormap(["#EF9A9A","#FFF59D","#90CAF9"])
	plt.contourf(x0,x1,zz,linewidth = 5,cmap = custom_cmap);
	w = model.coef_[0];
	b = model.intercept_[0];
	plot_x = np.linspace(axis[0],axis[1],200);
	up_y = (-w[0]/w[1])*plot_x-b/w[1]+1/w[1];
	down_y = (-w[0]/w[1])*plot_x-b/w[1]-1/w[1];
	up_index = (up_y >= axis[2]) & (up_y <=axis[3]);
	down_index = (down_y >= axis[2]) & (down_y <=axis[3]);
	plt.plot(plot_x[up_index],up_y[up_index],c='green');
	plt.plot(plot_x[down_index],down_y[down_index],c='green');
	
	
	
#数据可视化操作
plt.figure()
plot_svc_decision_boundary(svc,axis=[-3,3,-3,3]);
plt.scatter(x_strandard[y==0,0],x_strandard[y==0,1],color='red');
plt.scatter(x_strandard[y==1,0],x_strandard[y==1,1],color = 'blue');
plt.show();
