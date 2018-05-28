import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from plot_decision_boundary import plot_decision_boundary

#创建iris数据集
iris = datasets.load_iris();
x = iris.data[:,2:] 
y = iris.target

#DecisionTreeClassifier模拟
#dt_clf = DecisionTreeClassifier(max_depth = 2, criterion = "entropy");    #信息熵
dt_clf = DecisionTreeClassifier(max_depth = 2, criterion = "gini")        #基尼系数
dt_clf.fit(x,y)

#数据可视化
plt.figure();
plot_decision_boundary(dt_clf,axis=[0.5,7.5,0,3]);
plt.scatter(x[y==0,0],x[y==0,1]);
plt.scatter(x[y==1,0],x[y==1,1]);
plt.scatter(x[y==2,0],x[y==2,1]);
plt.show();

