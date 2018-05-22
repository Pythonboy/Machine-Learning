import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(666);
x = np.random.uniform(-3,3,size=(100,1));
y = 0.5*x**2+x+3+np.random.normal(0,1,size = (100,1));
x_ = np.hstack((x**2,x));
linearregression = LinearRegression();
linearregression.fit(x_,y);
y_predict = linearregression.predict(x_);
score = linearregression.score(x_,y);
print(score);
print(linearregression.coef_);
print(linearregression.intercept_);

x=x.reshape(-1);

plt.figure();
plt.scatter(x,y);
plt.plot(np.sort(x),y_predict[np.argsort(x)],'r');
plt.show();