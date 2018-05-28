import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

x,y = datasets.make_moons(n_samples = 500,noise = 0.3, random_state = 42);
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42);
RFC = RandomForestClassifier(n_estimators = 500,random_state = 666, oob_score = True, n_jobs = -1)
RFC.fit(x,y);
print(RFC.oob_score_);
ETC = ExtraTreesClassifier(n_estimators = 500,random_state=666, bootstrap = True, oob_score = True);
ETC.fit(x,y);
print(ETC.oob_score_);