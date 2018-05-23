from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
iris = datasets.load_iris();
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y);

lr = LogisticRegression();
lr.fit(x_train,y_train);
sc1 = lr.score(x_test,y_test);

lr_2 = LogisticRegression(multi_class = "multinomial", solver = "newton-cg");
lr_2.fit(x_train,y_train);
sc2 = lr_2.score(x_test,y_test);
print("socre_1:",sc1);
print("socre_2:",sc2);

log_reg = LogisticRegression();
ovr = OneVsRestClassifier(log_reg);
ovr.fit(x_train,y_train);
sc3 = ovr.score(x_test,y_test);

log_reg_1 = LogisticRegression();
ovo = OneVsOneClassifier(log_reg_1);
ovo.fit(x_train,y_train);
sc4 = ovo.score(x_test,y_test);
print("socre_3:",sc3);
print("score_4:",sc4);


