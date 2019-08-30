import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

from sklearn.externals import joblib

# User to enter a comma-separated list of csv files as input
NAG_Files = raw_input("Please enter a comma-separated list of csv files you'd like to gate\n")
Output_Path = raw_input("Please enter the path for the outputs\n")

# Creating a list of files and file handles that can be used to process the files
NAG_Files = NAG_Files.split(",")
NAG_File_Handle = [pd.read_csv(NAG_Files[i]) for i in range(len(NAG_Files))]

#Removing columns where there are no entries in the data
def Preprocessing(Files, File_Handle, Dataframe):
	for i in range(len(Files)):
		File_Handle[i].dropna(axis=1, how='all', inplace=True)
	for i in range(len(File_Handle)):
		Dataframe = pd.concat([Dataframe, File_Handle[i]], axis=0, sort=False, ignore_index=True)
	Dataframe['Type'] = 1
	return Dataframe

def Check_Attributes_Files(Files, File_Handle):
	for i in range(len(Files)):
		print("############################")
		print(Files[i])
		print(File_Handle[i].shape)
		print(File_Handle[i].columns)
		print("############################")

def Check_Attributes_Dataframe(Dataframe):
	print("############################")
	print(Dataframe.shape)
	print(Dataframe.columns)
	print(Dataframe.head())
	print("############################")

def Subset_Setup(Superset_Dataframe, Subset_Dataframe, Final_Dataframe):
	Not_Subset_Dataframe = pd.DataFrame()
	Not_Subset_Dataframe = pd.concat([Superset_Dataframe, Subset_Dataframe, Subset_Dataframe]).drop_duplicates(keep=False)
	Not_Subset_Dataframe['Type'] = 0
	Final_Dataframe = pd.concat([Not_Subset_Dataframe, Subset_Dataframe], axis=0, sort=False, ignore_index=True)
	return Final_Dataframe

def Basic_Classification(Dataframe, Metrics_File_Name):
	Metrics_File_Handle = open(Metrics_File_Name, 'w+')
	X = Dataframe.loc[:, Dataframe.columns != 'Type']
	y = Dataframe[['Type']]
	print(X.head())
	print(y.head())

	################################################
	# 				NAIVE BAYES		   			   #	   
	################################################
	NB  = joblib.load(Metrics_File_Name[:-4] + "_NB.pkl")
	y_pred = NB.predict(X)
	Metrics_File_Handle.write("############################\n")
	Metrics_File_Handle.write("NAIVE BAYES\n")
	Metrics_File_Handle.write(str(metrics.accuracy_score(y, y_pred)*100)+"\n")
	Metrics_File_Handle.write(str(confusion_matrix(y, y_pred))+"\n")
	Metrics_File_Handle.write(str(classification_report(y, y_pred))+"\n")
	Metrics_File_Handle.write("############################\n")

	################################################
	# 		DECISION TREES						   #	   
	################################################
	DT  = joblib.load(Metrics_File_Name[:-4] + "_DT.pkl")
	y_pred = DT.predict(X)
	Metrics_File_Handle.write("############################\n")
	Metrics_File_Handle.write("DECISION TREES\n")
	Metrics_File_Handle.write(str(metrics.accuracy_score(y, y_pred)*100)+"\n")
	Metrics_File_Handle.write(str(confusion_matrix(y, y_pred))+"\n")
	Metrics_File_Handle.write(str(classification_report(y, y_pred))+"\n")
	Metrics_File_Handle.write("############################\n")
	
	################################################
	# 		MULTI-CLASS LOGISTIC REGRESSION		   #	   
	################################################
	LR  = joblib.load(Metrics_File_Name[:-4] + "_LR.pkl")
	y_pred = LR.predict(X)
	Metrics_File_Handle.write("############################\n")
	Metrics_File_Handle.write("MULTI-CLASS LOGISTIC REGRESSION\n")
	Metrics_File_Handle.write(str(metrics.accuracy_score(y, y_pred)*100)+"\n")
	Metrics_File_Handle.write(str(confusion_matrix(y, y_pred))+"\n")
	Metrics_File_Handle.write(str(classification_report(y, y_pred))+"\n")
	Metrics_File_Handle.write("############################\n")

NAG = pd.DataFrame()
Check_Attributes_Files(NAG_Files, NAG_File_Handle)
NAG = Preprocessing(NAG_Files, NAG_File_Handle, NAG)
Check_Attributes_Dataframe(NAG)
Basic_Classification(NAWBC, "NA_WBC.txt")