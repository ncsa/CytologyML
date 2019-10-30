import json
import pandas as pd
import numpy as np
import logging, sys
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

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

"""View requirements.txt for information on version requirements"""
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def input_count(files):
    #length of first row = number of columns = number of Types
    #Find number of rows for each type --> number of rows in a given column
    o = open('input_count.txt', 'w')
    for i in files:
        o.write(str(i) + '\n')
        o.write("-------------------" + '\n')
    o.close()

#Functions to write classification checks to file
def write_to_file(f, classes, df, val):
    f = open(f,'w+')
    f.write(str(classes[val]) + '\n')
    f.write(str(df.shape) + '\n')
    f.write(str(df['Type'].unique()) + '\n')
    f.write(str(df.groupby('Type').size()) + '\n')
    f.write("-------------------" + '\n')
    f.close()
def add_to_file(f, classes, df, val):
    f = open(f,'a')
    f.write(str(classes[val]) + '\n')
    f.write(str(df.shape) + '\n')
    f.write(str(df['Type'].unique()) + '\n')
    f.write(str(df.groupby('Type').size()) + '\n')
    f.write("-------------------" + '\n')

def sanity_check(superset, subset):
    f = open("sanity_checks.txt", "a")
    #check intersections
    nonzero = pd.DataFrame()
    nonzero = superset.where(superset['Type'] != 0)
    zero = pd.DataFrame()
    zero = superset.where(superset['Type'] == 0)
    intersection_type1 = pd.merge(subset, nonzero, how='inner')
    intersection_type0 = pd.merge(subset, zero, how='inner')
    f.write("Intersection of type 1: " )
    f.write(str(len(intersection_type1)) + '\n')
    f.write("Intersection of type 0: ")
    f.write(str(len(intersection_type0)) + '\n')
    f.write("-----------" + '\n')
    f.close()

def confusion_matrix(superset,subset):
    f = open("confusion_matrix.txt", "a")
    true_pos = pd.DataFrame()
    false_pos = pd.DataFrame()
    false_neg = pd.DataFrame()

    #Everything of Type = 1 in the Superset is our predicted value. Subset will contain the known values
    prediction = superset.loc[superset['Type'] == 1]

    #converting the dataframes to lists, and then to sets (lists are unhashable). If it's an ndarray, directly convert it to a set
    predicted_list = (prediction.values).tolist()
    predicted_set = set(map(tuple, predicted_list))

    known_set = 0
    #Subset can either be type Dataframe or type np.ndarray
    if(type(subset) is np.ndarray):
        known_set =set(subset.flatten())
    else:
        known_list = (subset.values).tolist()
        known_set = set(map(tuple, known_list))

    #Use set intersection methods
    m = len(predicted_set)
    n = len(known_set)
    f.write("Length of Known: ")
    f.write(str(len(known_set)) + '\n')
    f.write("Length of Predicted: ")
    f.write(str(len(predicted_set)) + '\n')

    #True positives should be in both the subset and superset
    TP = predicted_set.intersection(known_set)
    f.write("LENGTH OF TP: ")
    f.write(str(len(TP)) + '\n')

    #True negatives will not be in either set (subset or superset)
    # TN = !superset_set.intersection(!subset_set)
    # f.write("Length of TN: ")
    # f.write(str(len(TN)) + '\n')

    #False positives will be in the subset but not in the superset
    FP = predicted_set - known_set
    f.write("LENGTH OF FP: ")
    f.write(str(len(FP)) + '\n')

    #False negatives will be in the superset but not the subset
    FN = known_set - predicted_set
    f.write("LENGTH OF FN: ")
    f.write(str(len(FN)) + '\n')

    f.write("------------" + '\n')
    f.close()

def read_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--method", help="Method to use")
    args = parser.parse_args()

    print( "Method {} ".format(
        args.method,
        ))
    return args.method

#Removing columns where there are no entries in the data
def Preprocessing(Files, File_Handle, Dataframe):
    """Preprocressing will remove columns where there are no entries and set the Type to 1."""
    dropped = 0
    for i in range(len(Files)):
        File_Handle[i].dropna(axis=1, how='all', inplace=True)
        for i in range(len(File_Handle)):
            Dataframe = pd.concat([Dataframe, File_Handle[i]], axis=0, sort=False, ignore_index=True)
            Dataframe['Type'] = 1
            #Below is a check for how many columns are dropped, if more than 10 are dropped the user is notified
            dropped = Dataframe.isnull().values.ravel().sum()
            if(dropped > 10):
                raise Exception ("Too many columns were dropped from the dataframe", num_dropped)
            return Dataframe

def Subset_Setup_New(Superset_Dataframe, Subset_Dataframe, Final_Dataframe):
    """Subset setup will categorize everything that is in Superset but not in Subset as Type 0.
    The two subsets will then be concatenated to create the full dataframe."""
    ds1 = set([tuple(line) for line in Superset_Dataframe.values])
    ds2 = set([tuple(line) for line in Subset_Dataframe.values])
    Not_Subset_Dataframe = ds1.difference(ds2)
    Not_Subset_Dataframe = pd.DataFrame(list(ds1.difference(ds2)))
    Not_Subset_Dataframe['Type'] = 0
    Final_Dataframe = pd.concat([Not_Subset_Dataframe, Subset_Dataframe], axis=0, sort=False, ignore_index=True)
    logging.debug("########################")
    logging.debug(Subset_Dataframe.shape[0])
    logging.debug(Subset_Dataframe.columns)
    logging.debug(Not_Subset_Dataframe.shape[0])
    logging.debug(Not_Subset_Dataframe.columns)
    logging.debug(Final_Dataframe.shape[0])
    logging.debug(Final_Dataframe.columns)
    logging.debug("########################")
    return Final_Dataframe

def Subset_Setup(Superset_Dataframe, Subset_Dataframe, Final_Dataframe):
    """Subset setup is removing duplicates between the superset and subset, and elements of the superset that are
    not a part of subset are assigned Type 0 and will then be concatenated with the subset to create the final dataframe."""
    Not_Subset_Dataframe = pd.DataFrame()
    Not_Subset_Dataframe = pd.concat([Superset_Dataframe, Subset_Dataframe]).drop_duplicates(keep=False)
    Not_Subset_Dataframe['Type'] = 0
    Final_Dataframe = pd.concat([Not_Subset_Dataframe, Subset_Dataframe], axis=0, sort=False, ignore_index=True)
    logging.debug(Subset_Dataframe.shape[0])
    logging.debug(Not_Subset_Dataframe.shape[0])
    logging.debug(Final_Dataframe.shape[0])
    return Final_Dataframe

def Basic_Classification(Dataframe, Metrics_File_Name, Metrics_Path, Models_Path):
    """Basic Classification is using naive bayes, decision trees, and logistic regression
     in order to train the machine to detect patterns in the data."""
    Metrics_File_Handle = open(Metrics_Path+Metrics_File_Name, 'w+')
    X = Dataframe.loc[:, Dataframe.columns != 'Type']
    X = X.loc[:, X.columns != 'Time']
    y = Dataframe[['Type']]
    logging.debug(Metrics_File_Name[:-4])
    logging.debug(X.shape)
    logging.debug(X.columns)
    logging.debug(y.shape)
    logging.debug(y.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train = X
    y_train = y

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


def main():
    a = open("confusion_matrix.txt", 'w')
    a.write("""ORDER:
    ANAG/NAG,
    NAGWBC/NAG,
    WBCCD45D/CD45D
    WBCCD45L/CD45L
    CD45DCD19CD10C/CD19CD10C
    CD45DCD34/CD34
    CD45LCD19PL/CD19PL
    CD45LCD19NL/CD19NL
    CD19PLKPB/KPB
    CD19PLLPB/LPB
    CD19NLCD3CD16T/CD3CD16T
    CD19NLNK/NK
    CD19NLNBNT/NBNT
    CD19NLT/T
    """)
    a.write("------------" + '\n')
    a.close()
    #Read the data inputs from a JSON file
    with open('files.json', 'r') as f:
        files_dict = json.load(f)

    # Creating a list of files and file handles that can be used to process the files
    A_Files = files_dict['A_Files'].split(",")
    A_File_Handle = [pd.read_csv(A_Files[i], header=0) for i in range(len(A_Files))]
    NAG_Files = files_dict['NAG_Files'].split(",")
    NAG_File_Handle = [pd.read_csv(NAG_Files[i]) for i in range(len(NAG_Files))]
    WBC_Files = files_dict['WBC_Files'].split(",")
    WBC_File_Handle = [pd.read_csv(WBC_Files[i]) for i in range(len(WBC_Files))]
    CD45D_Files = files_dict['CD45D_Files'].split(",")
    CD45D_File_Handle = [pd.read_csv(CD45D_Files[i]) for i in range(len(CD45D_Files))]
    CD45L_Files = files_dict['CD45L_Files'].split(",")
    CD45L_File_Handle = [pd.read_csv(CD45L_Files[i]) for i in range(len(CD45L_Files))]
    CD19CD10C_Files = files_dict['CD19CD10C_Files'].split(",")
    CD19CD10C_File_Handle = [pd.read_csv(CD19CD10C_Files[i]) for i in range(len(CD19CD10C_Files))]
    CD34_Files = files_dict['CD34_Files'].split(",")
    CD34_File_Handle = [pd.read_csv(CD34_Files[i]) for i in range(len(CD34_Files))]
    CD19PL_Files = files_dict['CD19PL_Files'].split(",")
    CD19PL_File_Handle = [pd.read_csv(CD19PL_Files[i]) for i in range(len(CD19PL_Files))]
    CD19NL_Files = files_dict['CD19NL_Files'].split(",")
    CD19NL_File_Handle = [pd.read_csv(CD19NL_Files[i]) for i in range(len(CD19NL_Files))]
    KPB_Files = files_dict['KPB_Files'].split(",")
    KPB_File_Handle = [pd.read_csv(KPB_Files[i]) for i in range(len(KPB_Files))]
    LPB_Files = files_dict['LPB_Files'].split(",")
    LPB_File_Handle = [pd.read_csv(LPB_Files[i]) for i in range(len(LPB_Files))]
    CD3CD16T_Files = files_dict['CD3CD16T_Files'].split(",")
    CD3CD16T_File_Handle = [pd.read_csv(CD3CD16T_Files[i]) for i in range(len(CD3CD16T_Files))]
    NK_Files = files_dict['NK_Files'].split(",")
    NK_File_Handle = [pd.read_csv(NK_Files[i]) for i in range(len(NK_Files))]
    NBNT_Files = files_dict['NBNT_Files'].split(",")
    NBNT_File_Handle = [pd.read_csv(NBNT_Files[i]) for i in range(len(NBNT_Files))]
    T_Files = files_dict['T_Files'].split(",")
    T_File_Handle = [pd.read_csv(T_Files[i]) for i in range(len(T_Files))]
    Metrics_Path = files_dict['Metrics_Path']
    Models_Path = files_dict['Models_Path']

    files = [A_File_Handle, NAG_File_Handle, WBC_File_Handle, CD45D_File_Handle, CD45L_File_Handle,
    CD19CD10C_File_Handle, CD34_File_Handle, CD19PL_File_Handle, CD19NL_File_Handle, KPB_File_Handle,
    LPB_File_Handle, CD3CD16T_File_Handle,NK_File_Handle, NBNT_File_Handle, T_File_Handle]

    #Create a list of all possible cell type classifications
    classes = ["ANAG", "NAGWBC", "WBCCD45D", "WBCCD45DL", "CD45DCD34", "CD45LCD19PL", "CD45LCD19NL",
    "CD19PLKPB", "CD19PLLPB", "CD19NLCD3CD16T", "CD19NLNK", "CD19NLT", "CD19NLNBNT"]
    val = 0
    input_count(files)

    #The below code is running the subset classification code on each subsequent file in order to clearly establish the file hierarchy and classification diagram
    A = pd.DataFrame()
    NAG = pd.DataFrame()
    A = Preprocessing(A_Files, A_File_Handle, A)
    print("A", len(A.index))

    NAG = Preprocessing(NAG_Files, NAG_File_Handle, NAG)
    print("NAG", len(NAG.index))

    ANAG = pd.DataFrame()
    ANAG = Subset_Setup(A, NAG, ANAG)
    print("ANAG",len(ANAG.index))

    #get count of each type
    print("ANAG TYPE 1",(ANAG['Type'] != 0).sum())
    print("ANAG TYPE 0",(ANAG['Type'] != 1).sum())
    sanity_check(ANAG,NAG)
    confusion_matrix(ANAG,NAG)


    Basic_Classification(ANAG, "A_NAG.txt", Metrics_Path, Models_Path)
    write_to_file("results.txt", classes, ANAG, val)


    WBC = pd.DataFrame()
    WBC = Preprocessing(WBC_Files, WBC_File_Handle, WBC)
    print("WBC", len(WBC.index))
    NAGWBC = pd.DataFrame()
    NAGWBC = Subset_Setup(NAG, WBC, NAGWBC)

    #get count of each type
    print("NAGWBC", len(NAGWBC.index))
    print("NAGWBC TYPE 1",(NAGWBC['Type'] != 0).sum())
    print("NAGWBC TYPE 0",(NAGWBC['Type'] != 1).sum())
    sanity_check(NAGWBC, WBC)
    confusion_matrix(NAGWBC,WBC)

    Basic_Classification(NAGWBC, "NAG_WBC.txt", Metrics_Path, Models_Path)
    val = 1
    add_to_file("results.txt", classes, NAGWBC, val)


    CD45D = pd.DataFrame()
    CD45L = pd.DataFrame()
    CD45D = Preprocessing(CD45D_Files, CD45D_File_Handle, CD45D)
    CD45L = Preprocessing(CD45L_Files, CD45L_File_Handle, CD45L)
    print("CD45D Preprocessing", len(CD45D.index))
    print("CD45L Preprocessing", len(CD45L.index))
    WBCCD45D = pd.DataFrame()
    WBCCD45L = pd.DataFrame()
    WBCCD45D = Subset_Setup(WBC, CD45D, WBCCD45D)
    WBCCD45L = Subset_Setup(WBC, CD45L, WBCCD45L)

    print("CD45D Subset", len(WBCCD45D.index))
    print("CD45L Subset", len(WBCCD45L.index))

    #get count of each type
    print("WBCCD45D TYPE 1",(WBCCD45D['Type'] != 0).sum())
    print("WBCCD45D TYPE 0",(WBCCD45D['Type'] != 1).sum())
    print("WBCCD45L TYPE 1",(WBCCD45L['Type'] != 0).sum())
    print("WBCCD45L TYPE 0",(WBCCD45L['Type'] != 1).sum())
    sanity_check(WBCCD45D, CD45D)
    confusion_matrix(WBCCD45D,CD45D)
    sanity_check(WBCCD45L, CD45L)
    confusion_matrix(WBCCD45L,CD45L)

    Basic_Classification(WBCCD45D, "WBC_CD45D.txt", Metrics_Path, Models_Path)
    val = 2
    add_to_file("results.txt", classes, WBCCD45D, val)
    Basic_Classification(WBCCD45L, "WBC_CD45L.txt", Metrics_Path, Models_Path)
    val = 3
    add_to_file("results.txt", classes, WBCCD45L, val)


    CD19CD10C = pd.DataFrame()
    CD34 = pd.DataFrame()
    CD19CD10C = Preprocessing(CD19CD10C_Files, CD19CD10C_File_Handle, CD19CD10C)
    CD34 = Preprocessing(CD34_Files, CD34_File_Handle, CD34)
    print("CD19CD10C Preprocessing", len(CD19CD10C.index))
    print("CD34 Preprocessing", len(CD34.index))
    CD45DCD19CD10C = pd.DataFrame()
    CD45DCD34 = pd.DataFrame()
    CD45DCD19CD10C = Subset_Setup(CD45D, CD19CD10C, CD45DCD19CD10C)
    CD45DCD34 = Subset_Setup(CD45D, CD34, CD45DCD34)
    print("CD45DCD19CD10C Subset Setup", len(CD45DCD19CD10C.index))
    print("CD45DCD34 Subset Setup", len(CD45DCD34.index))

    #get count of each type
    print("CD45DCD19CD10C TYPE 1",(CD45DCD19CD10C['Type'] != 0).sum())
    print("CD45DCD19CD10C TYPE 0",(CD45DCD19CD10C['Type'] != 1).sum())
    print("CD45DCD34 TYPE 1",(CD45DCD34['Type'] != 0).sum())
    print("CD45DCD34 TYPE 0",(CD45DCD34['Type'] != 1).sum())
    sanity_check(CD45DCD19CD10C,CD19CD10C)
    confusion_matrix(CD45DCD19CD10C,CD19CD10C)
    sanity_check(CD45DCD34, CD34)
    confusion_matrix(CD45DCD34,CD34)



    # Basic_Classification(CD45DCD19CD10C, "CD45D_CD19CD10C.txt", Metrics_Path, Models_Path)
    Basic_Classification(CD45DCD34, "CD45D_CD34.txt", Metrics_Path, Models_Path)
    val = 4
    add_to_file("results.txt", classes, CD45DCD34, val)


    CD19PL = pd.DataFrame()
    CD19NL = pd.DataFrame()
    CD19PL = Preprocessing(CD19PL_Files, CD19PL_File_Handle, CD19PL)
    CD19NL = Preprocessing(CD19NL_Files, CD19NL_File_Handle, CD19NL)
    print("CD19PL Preprocessing", len(CD19PL.index))
    print("CD19NL Preprocessing", len(CD19NL.index))
    CD45LCD19PL = pd.DataFrame()
    CD45LCD19NL = pd.DataFrame()
    CD45LCD19PL = Subset_Setup(CD45L, CD19PL, CD45LCD19PL)
    CD45LCD19NL = Subset_Setup(CD45L, CD19NL, CD45LCD19NL)
    print("CD45LCD19PL Subset Setup", len(CD45LCD19PL.index))
    print("CD45LCD19NL Subset Setup", len(CD45LCD19NL.index))

    #get count of each type
    print("CD45LCD19PL TYPE 1",(CD45LCD19PL['Type'] != 0).sum())
    print("CD45LCD19PL TYPE 0",(CD45LCD19PL['Type'] != 1).sum())
    print("CD45LCD19NL TYPE 1",(CD45LCD19NL['Type'] != 0).sum())
    print("CD45LCD19NL TYPE 0",(CD45LCD19NL['Type'] != 1).sum())
    sanity_check(CD45LCD19PL,CD19PL)
    confusion_matrix(CD45LCD19PL,CD19PL)
    sanity_check(CD45LCD19NL, CD19NL)
    confusion_matrix(CD45LCD19NL,CD19NL)

    Basic_Classification(CD45LCD19PL, "CD45L_CD19PL.txt", Metrics_Path, Models_Path)
    val = 5
    add_to_file("results.txt", classes, CD45LCD19PL, val)
    Basic_Classification(CD45LCD19NL, "CD45L_CD19NL.txt", Metrics_Path, Models_Path)
    val = 6
    add_to_file("results.txt", classes, CD45LCD19NL, val)

    KPB = pd.DataFrame()
    LPB = pd.DataFrame()
    KPB = Preprocessing(KPB_Files, KPB_File_Handle, KPB)
    LPB = Preprocessing(LPB_Files, LPB_File_Handle, LPB)
    print("KPB Preprocessing", len(KPB.index))
    print("LPB Preprocessing", len(LPB.index))
    CD19PLKPB = pd.DataFrame()
    CD19PLLPB = pd.DataFrame()
    CD19PLKPB = Subset_Setup(CD19PL, KPB, CD19PLKPB)
    CD19PLLPB = Subset_Setup(CD19PL, LPB, CD19PLLPB)
    print("KPB Subset Setup", len(CD19PLKPB.index))
    print("LPB Subset Setup", len(CD19PLLPB.index))

    #get count of each type
    print("CD19PLKPB TYPE 1",(CD19PLKPB['Type'] != 0).sum())
    print("CD19PLKPB TYPE 0",(CD19PLKPB['Type'] != 1).sum())
    print("CD19PLLPB TYPE 1",(CD19PLLPB['Type'] != 0).sum())
    print("CD19PLLPB TYPE 0",(CD19PLLPB['Type'] != 1).sum())
    sanity_check(CD19PLKPB,KPB)
    confusion_matrix(CD19PLKPB,KPB)
    sanity_check(CD19PLLPB, LPB)
    confusion_matrix(CD19PLLPB,LPB)

    Basic_Classification(CD19PLKPB, "CD19PL_KPB.txt", Metrics_Path, Models_Path)
    val = 7
    add_to_file("results.txt", classes, CD19PLKPB, val)
    Basic_Classification(CD19PLLPB, "CD19PL_LPB.txt", Metrics_Path, Models_Path)
    val = 8
    add_to_file("results.txt", classes, CD19PLLPB, val)

    CD3CD16T = pd.DataFrame()
    NK = pd.DataFrame()
    NBNT = pd.DataFrame()
    T = pd.DataFrame()
    CD3CD16T = Preprocessing(CD3CD16T_Files, CD3CD16T_File_Handle, CD3CD16T)
    NK = Preprocessing(NK_Files, NK_File_Handle, NK)
    NBNT = Preprocessing(NBNT_Files, NBNT_File_Handle, NBNT)
    T = Preprocessing(T_Files, T_File_Handle, T)
    print("CD3CD16T Preprocessing", len(CD3CD16T.index))
    print("NK Preprocessing", len(NK.index))
    print("NBNT Preprocessing", len(NBNT.index))
    print("T Preprocessing", len(T.index))
    CD19NLCD3CD16T = pd.DataFrame()
    CD19NLNK = pd.DataFrame()
    CD19NLNBNT = pd.DataFrame()
    CD19NLT = pd.DataFrame()
    CD19NLCD3CD16T = Subset_Setup(CD19NL, CD3CD16T, CD19NLCD3CD16T)
    CD19NLNK = Subset_Setup(CD19NL, NK, CD19NLNK)
    CD19NLNBNT = Subset_Setup(CD19NL, NBNT, CD19NLNBNT)
    CD19NLT = Subset_Setup(CD19NL, T, CD19NLT)
    print("CD19NLCD3CD16T Subset Setup", len(CD19NLCD3CD16T.index))
    print("CD19NLNK Subset Setup", len(CD19NLNK.index))
    print("CD19NLNBNT Subset Setup", len(CD19NLNBNT.index))
    print("CD19NLT Subset Setup", len(CD19NLT.index))

    #get count of each type
    print("CD19NLCD3CD16T TYPE 1",(CD19NLCD3CD16T['Type'] != 0).sum())
    print("CD19NLCD3CD16T TYPE 0",(CD19NLCD3CD16T['Type'] != 1).sum())
    print("CD19NLNK TYPE 1",(CD19NLNK['Type'] != 0).sum())
    print("CD19NLNK TYPE 0",(CD19NLNK['Type'] != 1).sum())
    print("CD19NLNBNT TYPE 1",(CD19NLNBNT['Type'] != 0).sum())
    print("CD19NLNBNT TYPE 0",(CD19NLNBNT['Type'] != 1).sum())
    print("CD19NLT TYPE 1",(CD19NLT['Type'] != 0).sum())
    print("CD19NLT TYPE 0",(CD19NLT['Type'] != 1).sum())

    sanity_check(CD19NLCD3CD16T,CD3CD16T)
    confusion_matrix(CD19NLCD3CD16T,CD3CD16T)
    sanity_check(CD19NLNK, NK)
    confusion_matrix(CD19NLNK,NK)
    sanity_check(CD19NLNBNT, NBNT)
    confusion_matrix(CD19NLNBNT,NBNT)
    sanity_check(CD19NLT, T)
    confusion_matrix(CD19NLT,T)


    Basic_Classification(CD19NLCD3CD16T, "CD19NL_CD3CD16T.txt", Metrics_Path, Models_Path)
    val = 9
    add_to_file("results.txt", classes, CD19NLCD3CD16T, val)
    Basic_Classification(CD19NLNK, "CD19NL_NK.txt", Metrics_Path, Models_Path)
    val = 10
    add_to_file("results.txt", classes, CD19NLNK, val)
    Basic_Classification(CD19NLT, "CD19NL_T.txt", Metrics_Path, Models_Path)
    val = 11
    add_to_file("results.txt", classes, CD19NLT, val)
    Basic_Classification(CD19NLNBNT, "CD19NL_NBNT.txt", Metrics_Path, Models_Path)
    val = 12
    add_to_file("results.txt", classes, CD19NLNBNT, val)


if __debug__:
    print('Debug ON')
else:
    print('Debug OFF')

if __name__== "__main__":
  main()
