from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
x = iris.data 
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2);
standardscaler = StandardScaler()
standardscaler.fit(X_train);
x_train = standardscaler.transform(X_train);
x_test = standardscaler.transform(X_test);
KNN = KNeighborsClassifier(n_neighbors = 4);
KNN.fit(x_train,y_train);
y_predict = KNN.predict(x_test);
score = accuracy_score(y_test,y_predict);
print(y_predict);
print(score);


