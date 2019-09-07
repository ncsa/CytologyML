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
A_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for all cells\n")
NAG_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for non-aggregate cells\n")
WBC_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for WBC cells\n")
CD45D_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for CD45Dim cells\n")
CD45L_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for CD45L cells\n")
CD19CD10C_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for CD19CD10C cells\n")
CD34_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for CD34 cells\n")
CD19PL_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for CD19+ cells\n")
CD19NL_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for CD19- cells\n")
KPB_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for Kappa+ B cells\n")
LPB_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for Lambda+ cells\n")
CD3CD16T_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for CD3CD16 T cells\n")
NK_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for NK cells\n")
NBNT_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for NonB NonT cells\n")
T_Files = raw_input("Please enter a comma-separated list of csv files you'd like to use for T cells\n")
Metrics_Path = raw_input("Please enter a path for the location where you'd like the metrics to be stored\n")
Models_Path = raw_input("Please enter a path for the location where you'd like the models to be stored\n")

# Creating a list of files and file handles that can be used to process the files
A_Files = A_Files.split(",")
A_File_Handle = [pd.read_csv(A_Files[i], header=0) for i in range(len(A_Files))]
NAG_Files = NAG_Files.split(",")
NAG_File_Handle = [pd.read_csv(NAG_Files[i]) for i in range(len(NAG_Files))]
WBC_Files = WBC_Files.split(",")
WBC_File_Handle = [pd.read_csv(WBC_Files[i]) for i in range(len(WBC_Files))]
CD45D_Files = CD45D_Files.split(",")
CD45D_File_Handle = [pd.read_csv(CD45D_Files[i]) for i in range(len(CD45D_Files))]
CD45L_Files = CD45L_Files.split(",")
CD45L_File_Handle = [pd.read_csv(CD45L_Files[i]) for i in range(len(CD45L_Files))]
CD19CD10C_Files = CD19CD10C_Files.split(",")
CD19CD10C_File_Handle = [pd.read_csv(CD19CD10C_Files[i]) for i in range(len(CD19CD10C_Files))]
CD34_Files = CD34_Files.split(",")
CD34_File_Handle = [pd.read_csv(CD34_Files[i]) for i in range(len(CD34_Files))]
CD19PL_Files = CD19PL_Files.split(",")
CD19PL_File_Handle = [pd.read_csv(CD19PL_Files[i]) for i in range(len(CD19PL_Files))]
CD19NL_Files = CD19NL_Files.split(",")
CD19NL_File_Handle = [pd.read_csv(CD19NL_Files[i]) for i in range(len(CD19NL_Files))]
KPB_Files = KPB_Files.split(",")
KPB_File_Handle = [pd.read_csv(KPB_Files[i]) for i in range(len(KPB_Files))]
LPB_Files = LPB_Files.split(",")
LPB_File_Handle = [pd.read_csv(LPB_Files[i]) for i in range(len(LPB_Files))]
CD3CD16T_Files = CD3CD16T_Files.split(",")
CD3CD16T_File_Handle = [pd.read_csv(CD3CD16T_Files[i]) for i in range(len(CD3CD16T_Files))]
NK_Files = NK_Files.split(",")
NK_File_Handle = [pd.read_csv(NK_Files[i]) for i in range(len(NK_Files))]
NBNT_Files = NBNT_Files.split(",")
NBNT_File_Handle = [pd.read_csv(NBNT_Files[i]) for i in range(len(NBNT_Files))]
T_Files = T_Files.split(",")
T_File_Handle = [pd.read_csv(T_Files[i]) for i in range(len(T_Files))]

#Removing columns where there are no entries in the data
def Preprocessing(Files, File_Handle, Dataframe):
	for i in range(len(Files)):
		File_Handle[i].dropna(axis=1, how='all', inplace=True)
	for i in range(len(File_Handle)):
		Dataframe = pd.concat([Dataframe, File_Handle[i]], axis=0, sort=False, ignore_index=True)
	Dataframe['Type'] = 1
	return Dataframe

def Subset_Setup_New(Superset_Dataframe, Subset_Dataframe, Final_Dataframe):
	ds1 = set([tuple(line) for line in Superset_Dataframe.values])
	ds2 = set([tuple(line) for line in Subset_Dataframe.values])
	Not_Subset_Dataframe = ds1.difference(ds2)
	Not_Subset_Dataframe = pd.DataFrame(list(ds1.difference(ds2)))
	Not_Subset_Dataframe['Type'] = 0
	Final_Dataframe = pd.concat([Not_Subset_Dataframe, Subset_Dataframe], axis=0, sort=False, ignore_index=True)
	print("########################")
	print(Subset_Dataframe.shape[0])
	print(Subset_Dataframe.columns)
	print(Not_Subset_Dataframe.shape[0])
	print(Not_Subset_Dataframe.columns)
	print(Final_Dataframe.shape[0])
	print(Final_Dataframe.columns)
	print("########################")
	return Final_Dataframe

def Subset_Setup(Superset_Dataframe, Subset_Dataframe, Final_Dataframe):
	Not_Subset_Dataframe = pd.DataFrame()
	Not_Subset_Dataframe = pd.concat([Superset_Dataframe, Subset_Dataframe]).drop_duplicates(keep=False)
	Not_Subset_Dataframe['Type'] = 0
	Final_Dataframe = pd.concat([Not_Subset_Dataframe, Subset_Dataframe], axis=0, sort=False, ignore_index=True)
	print(Subset_Dataframe.shape[0])
	print(Not_Subset_Dataframe.shape[0])
	print(Final_Dataframe.shape[0])
	return Final_Dataframe

def Basic_Classification(Dataframe, Metrics_File_Name, Metrics_Path, Models_Path):
	Metrics_File_Handle = open(Metrics_Path+Metrics_File_Name, 'w+')
	X = Dataframe.loc[:, Dataframe.columns != 'Type']
	X = X.loc[:, X.columns != 'Time']
	y = Dataframe[['Type']]
	print(Metrics_File_Name[:-4])
	print(X.shape)
	print(X.columns)
	print(y.shape)
	print(y.columns)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	################################################
	# 				NAIVE BAYES		   			   #	   
	################################################
	Gaussian_NB = GaussianNB()
	Gaussian_NB.fit(X_train, np.ravel(y_train))
	y_pred = Gaussian_NB.predict(X_test)
	Metrics_File_Handle.write("############################\n")
	Metrics_File_Handle.write("NAIVE BAYES\n")
	Metrics_File_Handle.write(str(metrics.accuracy_score(y_test, y_pred)*100)+"\n")
	Metrics_File_Handle.write(str(confusion_matrix(y_test, y_pred))+"\n")
	Metrics_File_Handle.write(str(classification_report(y_test, y_pred))+"\n")
	Metrics_File_Handle.write("############################\n")
	NB_File = Models_Path + Metrics_File_Name[:-4] + "_NB.pkl"
	joblib.dump(Gaussian_NB, NB_File) 

	################################################
	# 		DECISION TREES						   #	   
	################################################
	Decision_Tree = tree.DecisionTreeClassifier()
	Decision_Tree.fit(X_train, np.ravel(y_train))
	y_pred = Decision_Tree.predict(X_test)
	Metrics_File_Handle.write("############################\n")
	Metrics_File_Handle.write("DECISION TREES\n")
	Metrics_File_Handle.write(str(metrics.accuracy_score(y_test, y_pred)*100)+"\n")
	Metrics_File_Handle.write(str(confusion_matrix(y_test, y_pred))+"\n")
	Metrics_File_Handle.write(str(classification_report(y_test, y_pred))+"\n")
	Metrics_File_Handle.write("############################\n")
	DT_File = Models_Path + Metrics_File_Name[:-4] + "_DT.pkl"
	joblib.dump(Decision_Tree, DT_File) 
	
	################################################
	# 		MULTI-CLASS LOGISTIC REGRESSION		   #	   
	################################################
	Logistic_regression = LogisticRegression(solver = 'lbfgs')
	Logistic_regression.fit(X_train, np.ravel(y_train))
	y_pred = Logistic_regression.predict(X_test)
	score = Logistic_regression.score(X_test, y_test)
	Metrics_File_Handle.write("############################\n")
	Metrics_File_Handle.write("MULTI-CLASS LOGISTIC REGRESSION\n")
	Metrics_File_Handle.write(str(score*100)+"\n")
	Metrics_File_Handle.write(str(confusion_matrix(y_test, y_pred))+"\n")
	Metrics_File_Handle.write(str(classification_report(y_test, y_pred))+"\n")
	Metrics_File_Handle.write("############################\n")
	LR_File = Models_Path + Metrics_File_Name[:-4] + "_LR.pkl"
	joblib.dump(Logistic_regression, LR_File) 

A = pd.DataFrame()
NAG = pd.DataFrame()
A = Preprocessing(A_Files, A_File_Handle, A)
NAG = Preprocessing(NAG_Files, NAG_File_Handle, NAG)
ANAG = pd.DataFrame()
ANAG = Subset_Setup(A, NAG, ANAG)
Basic_Classification(ANAG, "A_NAG.txt", Metrics_Path, Models_Path)


WBC = pd.DataFrame()
WBC = Preprocessing(WBC_Files, WBC_File_Handle, WBC)
NAGWBC = pd.DataFrame()
NAGWBC = Subset_Setup(NAG, WBC, NAGWBC)
Basic_Classification(NAGWBC, "NAG_WBC.txt", Metrics_Path, Models_Path)


CD45D = pd.DataFrame()
CD45L = pd.DataFrame()
CD45D = Preprocessing(CD45D_Files, CD45D_File_Handle, CD45D)
CD45L = Preprocessing(CD45L_Files, CD45L_File_Handle, CD45L)
WBCCD45D = pd.DataFrame()
WBCCD45L = pd.DataFrame()
WBCCD45D = Subset_Setup(WBC, CD45D, WBCCD45D)
WBCCD45L = Subset_Setup(WBC, CD45L, WBCCD45L)
Basic_Classification(WBCCD45D, "WBC_CD45D.txt", Metrics_Path, Models_Path)
Basic_Classification(WBCCD45L, "WBC_CD45L.txt", Metrics_Path, Models_Path)

CD19CD10C = pd.DataFrame()
CD34 = pd.DataFrame()
CD19CD10C = Preprocessing(CD19CD10C_Files, CD19CD10C_File_Handle, CD19CD10C)
CD34 = Preprocessing(CD34_Files, CD34_File_Handle, CD34)
CD45DCD19CD10C = pd.DataFrame()
CD45DCD34 = pd.DataFrame()
CD45DCD19CD10C = Subset_Setup(CD45D, CD19CD10C, CD45DCD19CD10C)
CD45DCD34 = Subset_Setup(CD45D, CD34, CD45DCD34)
Basic_Classification(CD45DCD19CD10C, "CD45D_CD19CD10C.txt", Metrics_Path, Models_Path)
Basic_Classification(CD45DCD34, "CD45D_CD34.txt", Metrics_Path, Models_Path)


CD19PL = pd.DataFrame()
CD19NL = pd.DataFrame()
CD19PL = Preprocessing(CD19PL_Files, CD19PL_File_Handle, CD19PL)
CD19NL = Preprocessing(CD19NL_Files, CD19NL_File_Handle, CD19NL)
CD45LCD19PL = pd.DataFrame()
CD45LCD19NL = pd.DataFrame()
CD45LCD19PL = Subset_Setup(CD45L, CD19PL, CD45LCD19PL)
CD45LCD19NL = Subset_Setup(CD45L, CD19NL, CD45LCD19NL)
Basic_Classification(CD45LCD19PL, "CD45L_CD19PL.txt", Metrics_Path, Models_Path)
Basic_Classification(CD45LCD19NL, "CD45L_CD19NL.txt", Metrics_Path, Models_Path)

KPB = pd.DataFrame()
LPB = pd.DataFrame()
KPB = Preprocessing(KPB_Files, KPB_File_Handle, KPB)
LPB = Preprocessing(LPB_Files, LPB_File_Handle, LPB)
CD19PLKPB = pd.DataFrame()
CD19PLLPB = pd.DataFrame()
CD19PLKPB = Subset_Setup(CD19PL, KPB, CD19PLKPB)
CD19PLLPB = Subset_Setup(CD19PL, LPB, CD19PLLPB)
Basic_Classification(CD19PLKPB, "CD19PL_KPB.txt", Metrics_Path, Models_Path)
Basic_Classification(CD19PLLPB, "CD19PL_LPB.txt", Metrics_Path, Models_Path)

CD3CD16T = pd.DataFrame()
NK = pd.DataFrame()
NBNT = pd.DataFrame()
T = pd.DataFrame()
CD3CD16T = Preprocessing(CD3CD16T_Files, CD3CD16T_File_Handle, CD3CD16T)
NK = Preprocessing(NK_Files, NK_File_Handle, NK)
NBNT = Preprocessing(NBNT_Files, NBNT_File_Handle, NBNT)
T = Preprocessing(T_Files, T_File_Handle, T)
CD19NLCD3CD16T = pd.DataFrame()
CD19NLNK = pd.DataFrame()
CD19NLNBNT = pd.DataFrame()
CD19NLT = pd.DataFrame()
CD19NLCD3CD16T = Subset_Setup(CD19NL, CD3CD16T, CD19NLCD3CD16T)
CD19NLNK = Subset_Setup(CD19NL, NK, CD19NLNK)
CD19NLNBNT = Subset_Setup(CD19NL, NBNT, CD19NLNBNT)
CD19NLT = Subset_Setup(CD19NL, T, CD19NLT)
Basic_Classification(CD19NLCD3CD16T, "CD19NL_CD3CD16T.txt", Metrics_Path, Models_Path)
Basic_Classification(CD19NLNK, "CD19NL_NK.txt", Metrics_Path, Models_Path)
Basic_Classification(CD19NLT, "CD19NL_T.txt", Metrics_Path, Models_Path)
Basic_Classification(CD19NLNBNT, "CD19NL_NBNT.txt", Metrics_Path, Models_Path)
