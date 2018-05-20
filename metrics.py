import numpy as np
from math import sqrt

def Mean_Squared_Error(y_test,y_predict):
	return np.sum((y_predict-y_test)**2)/len(y_test);
	
def Root_Mean_Squared_Error(y_test,y_predict):
	return sqrt(Mean_Squared_Error(y_test,y_predict));
	
def Mean_Absolute_Error(y_test,y_predict):
	return np.sum(np.absolute(y_predict-y_test))/len(y_test);

def r2_score(y_test,y_predict):
	return 1-Mean_Squared_Error(y_test,y_predict)/np.var(y_test);