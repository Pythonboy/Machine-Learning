import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC         #支持向量机分类蒜贩
from sklearn.linear_model import LogisticRegression      #逻辑回归算法
from sklearn.tree import DecisionTreeClassifier        #决策树算法
from sklearn.metrics import accuracy_score       #分类算法准确度
from sklearn.ensemble import VotingClassifier          #集成学习
from sklearn.neighbors import KNeighborsClassifier        #KNN 算法
from sklearn.ensemble import BaggingClassifier         #放回取样集成学习
from sklearn.ensemble import AdaBoostClassifier          #boost集成学习
from sklearn.ensemble import GradientBoostingClassifier


x,y = datasets.make_moons(n_samples = 500,noise = 0.3, random_state = 42);
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42);

#逻辑回归分类：
lr = LogisticRegression();
lr.fit(x_train,y_train);
logisticregression_score = lr.score(x_test,y_test);

#决策树分类
dt_clf = DecisionTreeClassifier();
dt_clf.fit(x_train,y_train);
dt_clf_score = dt_clf.score(x_test,y_test);

#支持向量机分类
svm_clf = SVC();
svm_clf.fit(x_train,y_train);
svm_clf_score = svm_clf.score(x_test,y_test);

#KNN算法分类
knn_clf = KNeighborsClassifier();
knn_clf.fit(x_train,y_train);
knn_clf_score = knn_clf.score(x_test,y_test);

#自定义集成学习预测
y_predict1 = lr.predict(x_test);
y_predict2 = dt_clf.predict(x_test);
y_predict3 = svm_clf.predict(x_test);
y_predict4 = knn_clf.predict(x_test);
y_predict = np.array((y_predict1+y_predict2+y_predict3+y_predict4)>=3,dtype='int');
voting_score = accuracy_score(y_test,y_predict);

#Voting Classifier 集成学习
voting_clf_hard = VotingClassifier(estimators=[
	("log_clf",LogisticRegression()),
	("svm_clf",SVC()),
	("dt_clf",DecisionTreeClassifier(random_state = 666)),
	("knn_clf",KNeighborsClassifier())
	],voting = 'hard');
voting_clf_soft = VotingClassifier(estimators=[
	("log_clf",LogisticRegression()),
	("svm_clf",SVC(probability = True)),
	("dt_clf",DecisionTreeClassifier(random_state = 666)),
	("knn_clf",KNeighborsClassifier())
	],voting = 'soft');
voting_clf_hard.fit(x_train,y_train);
voting_clf_soft.fit(x_train,y_train);
sklearn_voting_clf_hard_score = voting_clf_hard.score(x_test,y_test);
sklearn_voting_clf_soft_score = voting_clf_soft.score(x_test,y_test)

#Bagging集成学习
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators = 500,max_samples = 100, bootstrap = True);
bagging_clf.fit(x_train,y_train);
bagging_clf_score = bagging_clf.score(x_test,y_test);

#Bagging集成学习 & 使用OOB
bagging_clf_1 = BaggingClassifier(DecisionTreeClassifier(),n_estimators = 500,max_samples = 100, bootstrap = True,oob_score = True);
bagging_clf_1.fit(x,y);
bagging_clf_score_ = bagging_clf_1.oob_score_

#Boosting集成学习
adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2),n_estimators = 500);
adaboost.fit(x_train,y_train);
adaboost_clf_score = adaboost.score(x_test,y_test);

gbc = GradientBoostingClassifier(max_depth = 2, n_estimators = 30);
gbc.fit(x_train,y_train);
gradientboostclassifier_socre = gbc.score(x_test,y_test);


#输出
print("lr_score:%f\ndt_clf_score:%f\nsvm_clf_score:%f\nknn_clf_score:%f\nvoting_score:%f\n"%(logisticregression_score,dt_clf_score,svm_clf_score,knn_clf_score,voting_score));
print("sklearn_voting_clf_score:\n hard:%f\n soft:%f"%(sklearn_voting_clf_hard_score,sklearn_voting_clf_soft_score));
print("sklearn_bagging_clf_score:\n normal:%f\n oob_score_:%f"%(bagging_clf_score,bagging_clf_score_));
print("sklearn_adaboost_clf_score:%f\nsklearn_gradientboostclassifier_score:%f"%(adaboost_clf_score,gradientboostclassifier_socre));

#数据可视化
#plt.figure(figsize=(10,10));
#plt.scatter(x[y==0,0],x[y==0,1]);
#plt.scatter(x[y==1,0],x[y==1,1]);
#plt.show();