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

import pickle
from sklearn.externals.joblib import dump, load

# User to enter a comma-separated list of csv files as input
Cytometry_Files = raw_input("Please enter the path of the .csv file with flow data\n")
Models_Path = raw_input("Please enter a path for the location where the models are stored\n")
A = pd.read_csv(Cytometry_Files)

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

A = A.drop(['Time'], axis=1)
A_NAG_DT = load(Models_Path+"A_NAG_DT.pkl", 'r')
A_NAG_LR = load(Models_Path+"A_NAG_LR.pkl", 'r')
A_NAG_NB = load(Models_Path+"A_NAG_NB.pkl", 'r')
A = A.loc[:, A.columns != 'Time']
y_DT = A_NAG_DT.predict(A)
y_LR = A_NAG_LR.predict(A)
y_NB = A_NAG_NB.predict(A)
y = y_DT | y_LR | y_NB
A['Type'] = y
NAG = A.loc[A['Type'] == 1]
print("NAG\t" + str(NAG.shape[0]))


################################
################################
################################
################################

NAG = NAG.drop(['Type'], axis=1)
NAG_WBC_DT = load(Models_Path+"NAG_WBC_DT.pkl", 'r')
NAG_WBC_LR = load(Models_Path+"NAG_WBC_LR.pkl", 'r')
NAG_WBC_NB = load(Models_Path+"NAG_WBC_NB.pkl", 'r')
y_DT = NAG_WBC_DT.predict(NAG)
y_LR = NAG_WBC_LR.predict(NAG)
y_NB = NAG_WBC_NB.predict(NAG)
y = y_DT | y_LR | y_NB
NAG['Type'] = y
WBC = NAG.loc[NAG['Type'] == 1]
print("WBC\t" + str(WBC.shape[0]))

################################
################################
################################
################################

WBC = WBC.drop(['Type'], axis=1)
CD45D = pd.DataFrame()
WBC_CD45D_DT = load(Models_Path+"WBC_CD45D_DT.pkl", 'r')
WBC_CD45D_LR = load(Models_Path+"WBC_CD45D_LR.pkl", 'r')
WBC_CD45D_NB = load(Models_Path+"WBC_CD45D_NB.pkl", 'r')
y_DT = WBC_CD45D_DT.predict(WBC)
y_LR = WBC_CD45D_LR.predict(WBC)
y_NB = WBC_CD45D_NB.predict(WBC)
y = y_DT | y_LR | y_NB
WBC['Type'] = y
CD45D = WBC.loc[WBC['Type'] == 1]
print("CD45D\t" + str(CD45D.shape[0]))

WBC = WBC.drop(['Type'], axis=1)
CD45L = pd.DataFrame()
WBC_CD45L_DT = load(Models_Path+"WBC_CD45L_DT.pkl", 'r')
WBC_CD45L_LR = load(Models_Path+"WBC_CD45L_LR.pkl", 'r')
WBC_CD45L_NB = load(Models_Path+"WBC_CD45L_NB.pkl", 'r')
y_DT = WBC_CD45L_DT.predict(WBC)
y_LR = WBC_CD45L_LR.predict(WBC)
y_NB = WBC_CD45L_NB.predict(WBC)
y = y_DT | y_LR | y_NB
WBC['Type'] = y
CD45L = WBC.loc[WBC['Type'] == 1]
print("CD45L\t" + str(CD45L.shape[0]))

################################
################################
################################
################################

CD45D = CD45D.drop(['Type'], axis=1)
CD19CD10C = pd.DataFrame()
CD45D_CD19CD10C_DT = load(Models_Path+"CD45D_CD19CD10C_DT.pkl", 'r')
CD45D_CD19CD10C_LR = load(Models_Path+"CD45D_CD19CD10C_LR.pkl", 'r')
CD45D_CD19CD10C_NB = load(Models_Path+"CD45D_CD19CD10C_NB.pkl", 'r')
y_DT = CD45D_CD19CD10C_DT.predict(CD45D)
y_LR = CD45D_CD19CD10C_LR.predict(CD45D)
y_NB = CD45D_CD19CD10C_NB.predict(CD45D)
y = y_DT | y_LR | y_NB
CD45D['Type'] = y
CD19CD10C = CD45D.loc[CD45D['Type'] == 1]
print("CD19CD10C\t" + str(CD19CD10C.shape[0]))

CD45D = CD45D.drop(['Type'], axis=1)
CD34 = pd.DataFrame()
CD45D_CD34_DT = load(Models_Path+"CD45D_CD34_DT.pkl", 'r')
CD45D_CD34_LR = load(Models_Path+"CD45D_CD34_LR.pkl", 'r')
CD45D_CD34_NB = load(Models_Path+"CD45D_CD34_NB.pkl", 'r')
y_DT = CD45D_CD34_DT.predict(CD45D)
y_LR = CD45D_CD34_LR.predict(CD45D)
y_NB = CD45D_CD34_NB.predict(CD45D)
y = y_DT | y_LR | y_NB
CD45D['Type'] = y
CD34 = CD45D.loc[CD45D['Type'] == 1]
print("CD34\t" + str(CD34.shape[0]))

################################
################################

CD45L = CD45L.drop(['Type'], axis=1)
CD19PL = pd.DataFrame()
CD45L_CD19PL_DT = load(Models_Path+"CD45L_CD19PL_DT.pkl", 'r')
CD45L_CD19PL_LR = load(Models_Path+"CD45L_CD19PL_LR.pkl", 'r')
CD45L_CD19PL_NB = load(Models_Path+"CD45L_CD19PL_NB.pkl", 'r')
y_DT = CD45L_CD19PL_DT.predict(CD45L)
y_LR = CD45L_CD19PL_LR.predict(CD45L)
y_NB = CD45L_CD19PL_NB.predict(CD45L)
y = y_DT | y_LR | y_NB
CD45L['Type'] = y
CD19PL = CD45L.loc[CD45L['Type'] == 1]
print("CD19PL\t" + str(CD19PL.shape[0]))

CD45L = CD45L.drop(['Type'], axis=1)
CD19NL = pd.DataFrame()
CD45L_CD19NL_DT = load(Models_Path+"CD45L_CD19NL_DT.pkl", 'r')
CD45L_CD19NL_LR = load(Models_Path+"CD45L_CD19NL_LR.pkl", 'r')
CD45L_CD19NL_NB = load(Models_Path+"CD45L_CD19NL_NB.pkl", 'r')
y_DT = CD45L_CD19NL_DT.predict(CD45L)
y_LR = CD45L_CD19NL_LR.predict(CD45L)
y_NB = CD45L_CD19NL_NB.predict(CD45L)
y = y_DT | y_LR | y_NB
CD45L['Type'] = y
CD19NL = CD45L.loc[CD45L['Type'] == 1]
print("CD19NL\t" + str(CD19NL.shape[0]))                                

################################
################################
################################
################################

CD19PL = CD19PL.drop(['Type'], axis=1)
KPB = pd.DataFrame()
CD19PL_KPB_DT = load(Models_Path+"CD19PL_KPB_DT.pkl", 'r')
CD19PL_KPB_LR = load(Models_Path+"CD19PL_KPB_LR.pkl", 'r')
CD19PL_KPB_NB = load(Models_Path+"CD19PL_KPB_NB.pkl", 'r')
y_DT = CD19PL_KPB_DT.predict(CD19PL)
y_LR = CD19PL_KPB_LR.predict(CD19PL)
y_NB = CD19PL_KPB_NB.predict(CD19PL)
y = y_DT | y_LR | y_NB
CD19PL['Type'] = y
KPB = CD19PL.loc[CD19PL['Type'] == 1]
print("KPB\t" + str(KPB.shape[0]))

CD19PL = CD19PL.drop(['Type'], axis=1)
LPB = pd.DataFrame()
CD19PL_LPB_DT = load(Models_Path+"CD19PL_LPB_DT.pkl", 'r')
CD19PL_LPB_LR = load(Models_Path+"CD19PL_LPB_LR.pkl", 'r')
CD19PL_LPB_NB = load(Models_Path+"CD19PL_LPB_NB.pkl", 'r')
y_DT = CD19PL_LPB_DT.predict(CD19PL)
y_LR = CD19PL_LPB_LR.predict(CD19PL)
y_NB = CD19PL_LPB_NB.predict(CD19PL)
y = y_DT | y_LR | y_NB
CD19PL['Type'] = y
LPB = CD19PL.loc[CD19PL['Type'] == 1]
print("LPB\t" + str(LPB.shape[0]))

################################
################################

CD19NL = CD19NL.drop(['Type'], axis=1)
CD3CD16T = pd.DataFrame()
CD19NL_CD3CD16T_DT = load(Models_Path+"CD19NL_CD3CD16T_DT.pkl", 'r')
CD19NL_CD3CD16T_LR = load(Models_Path+"CD19NL_CD3CD16T_LR.pkl", 'r')
CD19NL_CD3CD16T_NB = load(Models_Path+"CD19NL_CD3CD16T_NB.pkl", 'r')
y_DT = CD19NL_CD3CD16T_DT.predict(CD19NL)
y_LR = CD19NL_CD3CD16T_LR.predict(CD19NL)
y_NB = CD19NL_CD3CD16T_NB.predict(CD19NL)
y = y_DT | y_LR | y_NB
CD19NL['Type'] = y
CD3CD16T = CD19NL.loc[CD19NL['Type'] == 1]
print("CD3CD16T\t" + str(CD3CD16T.shape[0]))

CD19NL = CD19NL.drop(['Type'], axis=1)
NK = pd.DataFrame()
CD19NL_NK_DT = load(Models_Path+"CD19NL_NK_DT.pkl", 'r')
CD19NL_NK_LR = load(Models_Path+"CD19NL_NK_LR.pkl", 'r')
CD19NL_NK_NB = load(Models_Path+"CD19NL_NK_NB.pkl", 'r')
y_DT = CD19NL_NK_DT.predict(CD19NL)
y_LR = CD19NL_NK_LR.predict(CD19NL)
y_NB = CD19NL_NK_NB.predict(CD19NL)
y = y_DT | y_LR | y_NB
CD19NL['Type'] = y
NK = CD19NL.loc[CD19NL['Type'] == 1]
print("NK\t" + str(NK.shape[0]))

CD19NL = CD19NL.drop(['Type'], axis=1)
NBNT = pd.DataFrame()
CD19NL_NBNT_DT = load(Models_Path+"CD19NL_NBNT_DT.pkl", 'r')
CD19NL_NBNT_LR = load(Models_Path+"CD19NL_NBNT_LR.pkl", 'r')
CD19NL_NBNT_NB = load(Models_Path+"CD19NL_NBNT_NB.pkl", 'r')
y_DT = CD19NL_NBNT_DT.predict(CD19NL)
y_LR = CD19NL_NBNT_LR.predict(CD19NL)
y_NB = CD19NL_NBNT_NB.predict(CD19NL)
y = y_DT | y_LR | y_NB
CD19NL['Type'] = y
NBNT = CD19NL.loc[CD19NL['Type'] == 1]
print("NBNT\t" + str(NBNT.shape[0]))

CD19NL = CD19NL.drop(['Type'], axis=1)
T = pd.DataFrame()
CD19NL_T_DT = load(Models_Path+"CD19NL_T_DT.pkl", 'r')
CD19NL_T_LR = load(Models_Path+"CD19NL_T_LR.pkl", 'r')
CD19NL_T_NB = load(Models_Path+"CD19NL_T_NB.pkl", 'r')
y_DT = CD19NL_T_DT.predict(CD19NL)
y_LR = CD19NL_T_LR.predict(CD19NL)
y_NB = CD19NL_T_NB.predict(CD19NL)
y = y_DT | y_LR | y_NB
CD19NL['Type'] = y
T = CD19NL.loc[CD19NL['Type'] == 1]
print("T\t" + str(T.shape[0]))


################################
################################
################################
################################

