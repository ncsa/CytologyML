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
Output_Path = raw_input("Please enter the path for the outputs\n")
Models_Path = raw_input("Please enter the path for the models\n")

# Creating a list of files and file handles that can be used to process the files
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
	Metrics_File_Handle = open(Output_Path+Metrics_File_Name, 'w+')
	X = Dataframe.loc[:, Dataframe.columns != 'Type']
	y = Dataframe[['Type']]
	print(X.head())
	print(y.head())
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
	NB_File = Metrics_File_Name[:-4] + "_NB.pkl"
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
	DT_File = Metrics_File_Name[:-4] + "_DT.pkl"
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
	LR_File = Metrics_File_Name[:-4] + "_LR.pkl"
	joblib.dump(Logistic_regression, LR_File) 

	################################################
	# 				NEURAL NETWORKS	   			   #	   
	################################################
	#encoder = LabelEncoder()
	#encoder.fit(y)
	#y = encoder.transform(y)
	#y = np_utils.to_categorical(y)
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	#X_train = np.expand_dims(X_train, axis=2)
	#X_test = np.expand_dims(X_test, axis=2)
	#print(X_train.shape)
	#CNN_Classifier = Sequential()
	#CNN_Classifier.add(Conv1D(filters=32, kernel_size=5, input_shape=(15, 1)))
	#CNN_Classifier.add(MaxPooling1D(pool_size=5))
	#CNN_Classifier.add(Flatten())
	#CNN_Classifier.add(Dense(10, activation='relu'))
	#CNN_Classifier.add(Dense(2, activation='softmax'))
	#CNN_Classifier.compile(optimizer ='adam',loss='categorical_crossentropy', metrics =['accuracy'])
	#CNN_Classifier.fit(X_train,y_train, batch_size=1, epochs=10)
	#print(CNN_Classifier.summary())
	#scores = CNN_Classifier.evaluate(X_test, y_test)

	#y_pred = CNN_Classifier.predict(X_test)
	#y_pred = np.argmax(y_pred, axis=1)
	#y_test = np.argmax(y_test, axis=1)
	#print("############################")
	#print("CONVOLUTIONAL NEURAL NETWORK")
	#print("\n%s: %.2f%%" % (CNN_Classifier.metrics_names[1], scores[1]*100))
	#print(confusion_matrix(y_test, y_pred))
	#print(classification_report(y_test, y_pred))
	#print("############################")

	#NN_Classifier = Sequential()
	#NN_Classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=X.shape[1]))
	#NN_Classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
	#NN_Classifier.add(Dense(2, activation='softmax', kernel_initializer='random_normal'))
	#NN_Classifier.compile(optimizer ='adam',loss='categorical_crossentropy', metrics =['accuracy'])
	#NN_Classifier.fit(X_train,y_train, batch_size=10, epochs=10)
	#scores = NN_Classifier.evaluate(X_test, y_test)

	#y_pred = NN_Classifier.predict(X_test)
	#y_pred = np.argmax(y_pred, axis=1)
	#y_test = np.argmax(y_test, axis=1)
	#print("############################")
	#print("NEURAL NETWORKS")
	#print("\n%s: %.2f%%" % (NN_Classifier.metrics_names[1], scores[1]*100))
	#print(confusion_matrix(y_test, y_pred))
	#print(classification_report(y_test, y_pred))
	#print("############################")


NAG = pd.DataFrame()
WBC = pd.DataFrame()
Check_Attributes_Files(NAG_Files, NAG_File_Handle)
Check_Attributes_Files(WBC_Files, WBC_File_Handle)
NAG = Preprocessing(NAG_Files, NAG_File_Handle, NAG)
WBC = Preprocessing(WBC_Files, WBC_File_Handle, WBC)
Check_Attributes_Dataframe(NAG)
Check_Attributes_Dataframe(WBC)
NAWBC = pd.DataFrame()
NAWBC = Subset_Setup(NAG, WBC, NAWBC)
Check_Attributes_Dataframe(NAWBC)
Basic_Classification(NAWBC, "NA_WBC.txt")


CD45D = pd.DataFrame()
CD45L = pd.DataFrame()
Check_Attributes_Files(CD45D_Files, CD45D_File_Handle)
Check_Attributes_Files(CD45L_Files, CD45L_File_Handle)
CD45D = Preprocessing(CD45D_Files, CD45D_File_Handle, CD45D)
CD45L = Preprocessing(CD45L_Files, CD45L_File_Handle, CD45L)
Check_Attributes_Dataframe(CD45D)
Check_Attributes_Dataframe(CD45L)
WBCCD45D = pd.DataFrame()
WBCCD45L = pd.DataFrame()
WBCCD45D = Subset_Setup(WBC, CD45D, WBCCD45D)
WBCCD45L = Subset_Setup(WBC, CD45L, WBCCD45L)
Check_Attributes_Dataframe(CD45D)
Check_Attributes_Dataframe(CD45L)
Basic_Classification(WBCCD45D, "WBC_CD45D.txt")
Basic_Classification(WBCCD45L, "WBC_CD45L.txt")

CD19CD10C = pd.DataFrame()
CD34 = pd.DataFrame()
Check_Attributes_Files(CD19CD10C_Files, CD19CD10C_File_Handle)
Check_Attributes_Files(CD34_Files, CD34_File_Handle)
CD19CD10C = Preprocessing(CD19CD10C_Files, CD19CD10C_File_Handle, CD19CD10C)
CD34 = Preprocessing(CD34_Files, CD34_File_Handle, CD34)
Check_Attributes_Dataframe(CD19CD10C)
Check_Attributes_Dataframe(CD34)
CD45DCD19CD10C = pd.DataFrame()
CD45DCD34 = pd.DataFrame()
CD45DCD19CD10C = Subset_Setup(CD45D, CD19CD10C, CD45DCD19CD10C)
CD45DCD34 = Subset_Setup(CD45D, CD34, CD45DCD34)
Check_Attributes_Dataframe(CD19CD10C)
Check_Attributes_Dataframe(CD34)
Basic_Classification(CD45DCD19CD10C, "CD45D_CD19CD10C.txt")
Basic_Classification(CD45DCD34, "CD45D_CD34.txt")


CD19PL = pd.DataFrame()
CD19NL = pd.DataFrame()
Check_Attributes_Files(CD19PL_Files, CD19PL_File_Handle)
Check_Attributes_Files(CD19NL_Files, CD19NL_File_Handle)
CD19PL = Preprocessing(CD19PL_Files, CD19PL_File_Handle, CD19PL)
CD19NL = Preprocessing(CD19NL_Files, CD19NL_File_Handle, CD19NL)
Check_Attributes_Dataframe(CD19PL)
Check_Attributes_Dataframe(CD19NL)
CD45LCD19PL = pd.DataFrame()
CD45LCD19NL = pd.DataFrame()
CD45LCD19PL = Subset_Setup(CD45L, CD19PL, CD45LCD19PL)
CD45LCD19NL = Subset_Setup(CD45L, CD19NL, CD45LCD19NL)
Check_Attributes_Dataframe(CD19PL)
Check_Attributes_Dataframe(CD19NL)
Basic_Classification(CD45LCD19PL, "CD45L_CD19PL.txt")
Basic_Classification(CD45LCD19PL, "CD45L_CD19NL.txt")

KPB = pd.DataFrame()
LPB = pd.DataFrame()
Check_Attributes_Files(KPB_Files, KPB_File_Handle)
Check_Attributes_Files(LPB_Files, LPB_File_Handle)
KPB = Preprocessing(KPB_Files, KPB_File_Handle, KPB)
LPB = Preprocessing(LPB_Files, LPB_File_Handle, LPB)
Check_Attributes_Dataframe(KPB)
Check_Attributes_Dataframe(LPB)
CD19PLKPB = pd.DataFrame()
CD19PLLPB = pd.DataFrame()
CD19PLKPB = Subset_Setup(CD19PL, KPB, CD19PLKPB)
CD19PLLPB = Subset_Setup(CD19PL, LPB, CD19PLLPB)
Check_Attributes_Dataframe(KPB)
Check_Attributes_Dataframe(LPB)
Basic_Classification(CD19PLKPB, "CD19PL_KPB.txt")
Basic_Classification(CD19PLLPB, "CD19PL_LPB.txt")

CD3CD16T = pd.DataFrame()
NK = pd.DataFrame()
NBNT = pd.DataFrame()
T = pd.DataFrame()
Check_Attributes_Files(CD3CD16T_Files, CD3CD16T_File_Handle)
Check_Attributes_Files(NK_Files, NK_File_Handle)
Check_Attributes_Files(NBNT_Files, NBNT_File_Handle)
Check_Attributes_Files(T_Files, T_File_Handle)
CD3CD16T = Preprocessing(CD3CD16T_Files, CD3CD16T_File_Handle, CD3CD16T)
NK = Preprocessing(NK_Files, NK_File_Handle, NK)
NBNT = Preprocessing(NBNT_Files, NBNT_File_Handle, NBNT)
T = Preprocessing(T_Files, T_File_Handle, T)
Check_Attributes_Dataframe(CD3CD16T)
Check_Attributes_Dataframe(NK)
Check_Attributes_Dataframe(NBNT)
Check_Attributes_Dataframe(T)
CD19NLCD3CD16T = pd.DataFrame()
CD19NLNK = pd.DataFrame()
CD19NLNBNT = pd.DataFrame()
CD19NLT = pd.DataFrame()
CD19NLCD3CD16T = Subset_Setup(CD19NL, CD3CD16T, CD19NLCD3CD16T)
CD19NLNK = Subset_Setup(CD19NL, NK, CD19NLNK)
CD19NLNBNT = Subset_Setup(CD19NL, NBNT, CD19NLNBNT)
CD19NLT = Subset_Setup(CD19NL, T, CD19NLT)
Check_Attributes_Dataframe(CD3CD16T)
Check_Attributes_Dataframe(NK)
Check_Attributes_Dataframe(NBNT)
Check_Attributes_Dataframe(T)
Basic_Classification(CD19NLCD3CD16T, "CD19NL_CD3CD16T.txt")
Basic_Classification(CD19NLNK, "CD19NL_NK.txt")
Basic_Classification(CD19NLT, "CD19NL_T.txt")
Basic_Classification(CD19NLNBNT, "CD19NL_NBNT.txt")
