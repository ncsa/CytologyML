import pandas as pd
import numpy as np
import json
import argparse
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

import pickle
import sklearn.externals.joblib.numpy_pickle
from sklearn.externals.joblib import dump, load

def method_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--method", help="Method to use")
    args = parser.parse_args()
    print( "Method Being Used: {} ".format(
        args.method,
        ))
    return args.method

def Check_Attributes_Dataframe(Dataframe):
	"""Check the attributes of the dataframe to check that everything is included."""
	print("############################")
	print(Dataframe.shape)
	print(Dataframe.columns)
	print(Dataframe.head())
	print("############################")

def Subset_Setup(Superset_Dataframe, Subset_Dataframe, Final_Dataframe):
	"""Subset setup is classifying the breakdowns in separate subsets while maintaining the original grouping at each level of classification."""
	Not_Subset_Dataframe = pd.DataFrame()
	Not_Subset_Dataframe = pd.concat([Superset_Dataframe, Subset_Dataframe, Subset_Dataframe]).drop_duplicates(keep=False)
	Not_Subset_Dataframe['Type'] = 0
	Final_Dataframe = pd.concat([Not_Subset_Dataframe, Subset_Dataframe], axis=0, sort=False, ignore_index=True)
	return Final_Dataframe

def main():

	#Read the data inputs from the JSON file
	with open('files.json', 'r') as f:
		files_dict = json.load(f)

	Cytometry_Files = files_dict['A_Files']
	Models_Path = files_dict['Models_Path']
	A = pd.read_csv(Cytometry_Files)


	A = A.drop(['Time'], axis=1)
	A_NAG_DT = load(Models_Path+"A_NAG_DT.pkl", 'r')
	A_NAG_LR = load(Models_Path+"A_NAG_LR.pkl", 'r')
	A_NAG_NB = load(Models_Path+"A_NAG_NB.pkl", 'r')
	A = A.loc[:, A.columns != 'Time']
	y_DT = A_NAG_DT.predict(A)
	y_LR = A_NAG_LR.predict(A)
	y_NB = A_NAG_NB.predict(A)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	y = method_input()
	A['Type'] = y
	NAG_predicted = A.loc[A['Type'] == 1]
	print("NAG\t" + str(NAG_predicted.shape[0]))


	################################
	################################
	################################
	################################

	NAG_predicted = NAG_predicted.drop(['Type'], axis=1)
	NAG_WBC_DT = load(Models_Path+"NAG_WBC_DT.pkl", 'r')
	NAG_WBC_LR = load(Models_Path+"NAG_WBC_LR.pkl", 'r')
	NAG_WBC_NB = load(Models_Path+"NAG_WBC_NB.pkl", 'r')
	y_DT = NAG_WBC_DT.predict(NAG_predicted)
	y_LR = NAG_WBC_LR.predict(NAG_predicted)
	y_NB = NAG_WBC_NB.predict(NAG_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	NAG_predicted['Type'] = y
	#Can also say nag[type] = y_or or y_and or y_dt, etc...
	#make a function so that in the json you provide a method type: (user input) and depending on that your y becomes that value

	WBC_predicted = NAG_predicted.loc[NAG_predicted['Type'] == 1]
	print("WBC\t" + str(WBC_predicted.shape[0]))

	################################
	################################
	################################
	################################

	WBC_predicted = WBC_predicted.drop(['Type'], axis=1)
	CD45D_predicted = pd.DataFrame()
	WBC_CD45D_DT = load(Models_Path+"WBC_CD45D_DT.pkl", 'r')
	WBC_CD45D_LR = load(Models_Path+"WBC_CD45D_LR.pkl", 'r')
	WBC_CD45D_NB = load(Models_Path+"WBC_CD45D_NB.pkl", 'r')
	y_DT = WBC_CD45D_DT.predict(WBC_predicted)
	y_LR = WBC_CD45D_LR.predict(WBC_predicted)
	y_NB = WBC_CD45D_NB.predict(WBC_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	WBC_predicted['Type'] = y
	CD45D_predicted = WBC_predicted.loc[WBC_predicted['Type'] == 1]
	print("CD45D\t" + str(CD45D_predicted.shape[0]))

	WBC_predicted = WBC_predicted.drop(['Type'], axis=1)
	CD45L_predicted = pd.DataFrame()
	WBC_CD45L_DT = load(Models_Path+"WBC_CD45L_DT.pkl", 'r')
	WBC_CD45L_LR = load(Models_Path+"WBC_CD45L_LR.pkl", 'r')
	WBC_CD45L_NB = load(Models_Path+"WBC_CD45L_NB.pkl", 'r')
	y_DT = WBC_CD45L_DT.predict(WBC_predicted)
	y_LR = WBC_CD45L_LR.predict(WBC_predicted)
	y_NB = WBC_CD45L_NB.predict(WBC_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	WBC_predicted['Type'] = y
	CD45L = WBC_predicted.loc[WBC_predicted['Type'] == 1]
	print("CD45L\t" + str(CD45L_predicted.shape[0]))

	################################
	################################
	################################
	################################

	CD45D_predicted = CD45D_predicted.drop(['Type'], axis=1)
	CD19CD10C_predicted = pd.DataFrame()
	CD45D_CD19CD10C_DT = load(Models_Path+"CD45D_CD19CD10C_DT.pkl", 'r')
	 # CD45D_CD19CD10C_LR = load(Models_Path+"CD45D_CD19CD10C_LR.pkl", 'r')
	CD45D_CD19CD10C_NB = load(Models_Path+"CD45D_CD19CD10C_NB.pkl", 'r')
	y_DT = CD45D_CD19CD10C_DT.predict(CD45D_predicted)
	#y_LR = CD45D_CD19CD10C_LR.predict(CD45D)
	y_NB = CD45D_CD19CD10C_NB.predict(CD45D_predicted)
	#y = y_DT | y_LR | y_NB
	y_or = y_DT | y_NB
	y_and = y_DT & y_NB

	CD45D_predicted['Type'] = y
	CD19CD10C_predicted = CD45D_predicted.loc[CD45D_predicted['Type'] == 1]
	print("CD19CD10C\t" + str(CD19CD10C_predicted.shape[0]))

	CD45D_predicted = CD45D_predicted.drop(['Type'], axis=1)
	CD34_predicted = pd.DataFrame()
	CD45D_CD34_DT = load(Models_Path+"CD45D_CD34_DT.pkl", 'r')
	CD45D_CD34_LR = load(Models_Path+"CD45D_CD34_LR.pkl", 'r')
	CD45D_CD34_NB = load(Models_Path+"CD45D_CD34_NB.pkl", 'r')
	y_DT = CD45D_CD34_DT.predict(CD45D_predicted)
	y_LR = CD45D_CD34_LR.predict(CD45D_predicted)
	y_NB = CD45D_CD34_NB.predict(CD45D_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD45D_predicted['Type'] = y
	CD34_predicted = CD45D_predicted.loc[CD45D_predicted['Type'] == 1]
	print("CD34\t" + str(CD34_predicted.shape[0]))

	################################
	################################

	CD45L_predicted = CD45L_predicted.drop(['Type'], axis=1)
	CD19PL_predicted = pd.DataFrame()
	CD45L_CD19PL_DT = load(Models_Path+"CD45L_CD19PL_DT.pkl", 'r')
	CD45L_CD19PL_LR = load(Models_Path+"CD45L_CD19PL_LR.pkl", 'r')
	CD45L_CD19PL_NB = load(Models_Path+"CD45L_CD19PL_NB.pkl", 'r')
	y_DT = CD45L_CD19PL_DT.predict(CD45L_predicted)
	y_LR = CD45L_CD19PL_LR.predict(CD45L_predicted)
	y_NB = CD45L_CD19PL_NB.predict(CD45L_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD45L_predicted['Type'] = y
	CD19PL_predicted = CD45L_predicted.loc[CD45L_predicted['Type'] == 1]
	print("CD19PL\t" + str(CD19PL_predicted.shape[0]))

	CD45L_predicted = CD45L_predicted.drop(['Type'], axis=1)
	CD19NL_predicted = pd.DataFrame()
	CD45L_CD19NL_DT = load(Models_Path+"CD45L_CD19NL_DT.pkl", 'r')
	CD45L_CD19NL_LR = load(Models_Path+"CD45L_CD19NL_LR.pkl", 'r')
	CD45L_CD19NL_NB = load(Models_Path+"CD45L_CD19NL_NB.pkl", 'r')
	y_DT = CD45L_CD19NL_DT.predict(CD45L_predicted)
	y_LR = CD45L_CD19NL_LR.predict(CD45L_predicted)
	y_NB = CD45L_CD19NL_NB.predict(CD45L_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD45L_predicted['Type'] = y
	CD19NL_predicted = CD45L_predicted.loc[CD45L_predicted['Type'] == 1]
	print("CD19NL\t" + str(CD19NL_predicted.shape[0]))

	################################
	################################
	################################
	################################

	CD19PL_predicted = CD19PL_predicted.drop(['Type'], axis=1)
	KPB_predicted = pd.DataFrame()
	CD19PL_KPB_DT = load(Models_Path+"CD19PL_KPB_DT.pkl", 'r')
	CD19PL_KPB_LR = load(Models_Path+"CD19PL_KPB_LR.pkl", 'r')
	CD19PL_KPB_NB = load(Models_Path+"CD19PL_KPB_NB.pkl", 'r')
	y_DT = CD19PL_KPB_DT.predict(CD19PL_predicted)
	y_LR = CD19PL_KPB_LR.predict(CD19PL_predicted)
	y_NB = CD19PL_KPB_NB.predict(CD19PL_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD19PL_predicted['Type'] = y
	KPB_predicted = CD19PL_predicted.loc[CD19PL_predicted['Type'] == 1]
	print("KPB\t" + str(KPB_predicted.shape[0]))

	CD19PL_predicted = CD19PL_predicted.drop(['Type'], axis=1)
	LPB_predicted = pd.DataFrame()
	CD19PL_LPB_DT = load(Models_Path+"CD19PL_LPB_DT.pkl", 'r')
	CD19PL_LPB_LR = load(Models_Path+"CD19PL_LPB_LR.pkl", 'r')
	CD19PL_LPB_NB = load(Models_Path+"CD19PL_LPB_NB.pkl", 'r')
	y_DT = CD19PL_LPB_DT.predict(CD19PL_predicted)
	y_LR = CD19PL_LPB_LR.predict(CD19PL_predicted)
	y_NB = CD19PL_LPB_NB.predict(CD19PL_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD19PL_predicted['Type'] = y
	LPB_predicted = CD19PL_predicted.loc[CD19PL_predicted['Type'] == 1]
	print("LPB\t" + str(LPB_predicted.shape[0]))

	################################
	################################

	CD19NL_predicted = CD19NL_predicted.drop(['Type'], axis=1)
	CD3CD16T_predicted = pd.DataFrame()
	CD19NL_CD3CD16T_DT = load(Models_Path+"CD19NL_CD3CD16T_DT.pkl", 'r')
	CD19NL_CD3CD16T_LR = load(Models_Path+"CD19NL_CD3CD16T_LR.pkl", 'r')
	CD19NL_CD3CD16T_NB = load(Models_Path+"CD19NL_CD3CD16T_NB.pkl", 'r')
	y_DT = CD19NL_CD3CD16T_DT.predict(CD19NL_predicted)
	y_LR = CD19NL_CD3CD16T_LR.predict(CD19NL_predicted)
	y_NB = CD19NL_CD3CD16T_NB.predict(CD19NL_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD19NL_predicted['Type'] = y
	CD3CD16T_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
	print("CD3CD16T\t" + str(CD3CD16T_predicted.shape[0]))

	CD19NL_predicted = CD19NL_predicted.drop(['Type'], axis=1)
	NK_predicted = pd.DataFrame()
	CD19NL_NK_DT = load(Models_Path+"CD19NL_NK_DT.pkl", 'r')
	CD19NL_NK_LR = load(Models_Path+"CD19NL_NK_LR.pkl", 'r')
	CD19NL_NK_NB = load(Models_Path+"CD19NL_NK_NB.pkl", 'r')
	y_DT = CD19NL_NK_DT.predict(CD19NL_predicted)
	y_LR = CD19NL_NK_LR.predict(CD19NL_predicted)
	y_NB = CD19NL_NK_NB.predict(CD19NL_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD19NL_predicted['Type'] = y
	NK_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
	print("NK\t" + str(NK_predicted.shape[0]))

	CD19NL_predicted = CD19NL_predicted.drop(['Type'], axis=1)
	NBNT_predicted = pd.DataFrame()
	CD19NL_NBNT_DT = load(Models_Path+"CD19NL_NBNT_DT.pkl", 'r')
	CD19NL_NBNT_LR = load(Models_Path+"CD19NL_NBNT_LR.pkl", 'r')
	CD19NL_NBNT_NB = load(Models_Path+"CD19NL_NBNT_NB.pkl", 'r')
	y_DT = CD19NL_NBNT_DT.predict(CD19NL_predicted)
	y_LR = CD19NL_NBNT_LR.predict(CD19NL_predicted)
	y_NB = CD19NL_NBNT_NB.predict(CD19NL_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD19NL_predicted['Type'] = y
	NBNT_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
	print("NBNT\t" + str(NBNT_predicted.shape[0]))

	CD19NL_predicted = CD19NL_predicted.drop(['Type'], axis=1)
	T_predicted = pd.DataFrame()
	CD19NL_T_DT = load(Models_Path+"CD19NL_T_DT.pkl", 'r')
	CD19NL_T_LR = load(Models_Path+"CD19NL_T_LR.pkl", 'r')
	CD19NL_T_NB = load(Models_Path+"CD19NL_T_NB.pkl", 'r')
	y_DT = CD19NL_T_DT.predict(CD19NL_predicted)
	y_LR = CD19NL_T_LR.predict(CD19NL_predicted)
	y_NB = CD19NL_T_NB.predict(CD19NL_predicted)
	y_or = y_DT | y_LR | y_NB
	y_and = y_DT & y_LR & y_NB

	CD19NL_predicted['Type'] = y
	T_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
	print("T\t" + str(T_predicted.shape[0]))


	################################
	################################
	################################
	################################


if __name__== "__main__":
  main()
