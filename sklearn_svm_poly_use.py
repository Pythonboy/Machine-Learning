import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

#获得数据集
x,y = datasets.make_moons(noise=0.15,random_state=666);

#使用多项式特征的SVM
def PolynomialSVM(degree,C=1.0):
	return Pipeline([
	("poly",PolynomialFeatures(degree=degree)),
	("std_scaler",StandardScaler()),
	("linearSVC",LinearSVC(C=C))
	]);
poly_svc = PolynomialSVM(3);
poly_svc.fit(x,y);

#使用多项式核函数的SVM
def PolynomialKernelSVC(degree,C=1.0):
	return Pipeline([
		("std_scaler",StandardScaler()),
		("KernelSVC",SVC(kernel = "poly",degree = degree,C=C))
		]);
poly_kernel_svc = PolynomialKernelSVC(degree=3);
poly_kernel_svc.fit(x,y);

#高斯核函数
def gaussian(x,l):
	gamma = 1.0;
	return np.exp(-gamma*(x-l)**2);
def RBFKernelSVC(gamma=1.0):
	return Pipeline([
		("std_scaler",StandardScaler()),
		("KernelSVC",SVC(kernel = "rbf",gamma = gamma))
		]);	
rbf_svc = RBFKernelSVC(gamma=0.01);
rbf_svc.fit(x,y);
		
		
#绘制决策边界
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


#数据可视化
plt.figure();
plot_decision_boundary(rbf_svc,axis=[-1.5,2.5,-1.0,1.5]);
plt.scatter(x[y==0,0],x[y==0,1]);
plt.scatter(x[y==1,0],x[y==1,1]);
plt.show();


#    python sklearn_svm_poly_use.py