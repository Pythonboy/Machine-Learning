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

def r2_score(y_test,y_predict):
	return 1-Mean_Squared_Error(y_test,y_predict)/np.var(y_test);

def accuracy_score(y_test,y_predict):
	return 1.0*np.sum(y_test==y_predict)/len(y_test);
	
def TN(y_true,y_predict):
	assert len(y_true)==len(y_predict);
	return np.sum((y_true==0) & (y_predict==0));
	
def FN(y_true,y_predict):
	assert len(y_true)==len(y_predict);
	return np.sum((y_true==1) & (y_predict==0));

def TP(y_true,y_predict):
	assert len(y_true)==len(y_predict);
	return np.sum((y_true==1) & (y_predict==1));
	
def FP(y_true,y_predict):
	assert len(y_true)==len(y_predict);
	return np.sum((y_true==0) & (y_predict==1));
	
def Confusion_Matrix(y_true,y_predict):
	assert len(y_true)==len(y_predict);
	return np.array([
	[TN(y_true,y_predict),FP(y_true,y_predict)],
	[FN(y_true,y_predict),TP(y_true,y_predict)]
	])
	
def precision_score(y_true,y_predict):
	assert len(y_true)==len(y_predict);
	try:
		return TP(y_true,y_predict)/(FP(y_true,y_predict)+TP(y_true,y_predict));
	except:
		return 0;
		
def recall_score(y_true,y_predict):
	assert len(y_true)==len(y_predict);
	try:
		return TP(y_true,y_predict)/(FN(y_true,y_predict)+TP(y_true,y_predict));
	except:
		return 0;
def F1_score(self,y_test，y_predict):
	pre_sco = precision_score(y_test，y_predict);
	rec_sco = recall_score(y_test,y_predict);
	try:
		return 2.0*pre_sco*rec_sco/(pre_sco+rec_sco);
	except:
		return 0;
