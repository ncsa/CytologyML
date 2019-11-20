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
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

"""View requirements.txt for information on version requirements"""
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

#Functions to write classification checks to file
def add_to_file(f, classes, df, val):
    f = open(f,'a')
    f.write(str(classes[val]) + '\n')
    f.write(str(df.shape) + '\n')
    f.write(str(df['Type'].unique()) + '\n')
    f.write(str(df.groupby('Type').size()) + '\n')
    f.write("-------------------" + '\n')

def sanity_check(superset, subset):
    f = open("sanity_checks.txt", "a")
    #check intersections of subset with the nonzero and zero values from the superset
    nonzero = superset.loc[superset['Type'] == 1]
    zero = superset.loc[superset['Type'] == 0]
    intersection_type1 = pd.merge(subset, nonzero, how='inner')
    #intersection_type0 should be equal to 0: nothing of type 0 should be in the subset
    intersection_type0 = pd.merge(subset, zero, how='inner')
    f.write("count type 1 in subset: " + str(len(intersection_type1)) + '\n')
    f.write("count type 0 in subset: " + str(len(intersection_type0)) + '\n')

    #extra check: check that size of subset, size of nonzero, size of intersection_type1 are equal
    subset_size = len(subset)
    nonzero_size = len(nonzero)
    type1_size = len(intersection_type1)
    f.write("Size of subset: {} " + str(subset_size) + '\n')
    f.write("Size of nonzero entries extracted from superset: " + str(nonzero_size) + '\n')
    f.write("Size of intersection_type1 is: " + str(len(intersection_type1)) + '\n')
    if((nonzero_size == subset_size) and (nonzero_size == type1_size) and (subset_size == type1_size)):
        f.write("Size of the above 3 matrices are the same" +  '\n')
    else:
        f.write("Size of the above 3 matrices are NOT the same" +  '\n')
    f.write("-----------" + '\n')
    f.close()

#Removing columns where there are no entries in the data
def Preprocessing(Files, File_Handle, Dataframe):
    """Preprocressing will remove columns where there are no entries and set the Type to 1."""
    dropped = 0
    for i in range(len(Files)):
        File_Handle[i].dropna(axis=1, how='all', inplace=True)
        for i in range(len(Files)):
            Dataframe = pd.concat([Dataframe, File_Handle[i]], axis=0, sort=False, ignore_index=True)
            Dataframe['Type'] = 1
            #Below is a check for how many columns are dropped, if more than 5% of the dataframe was dropped, the user is notified
            dropped = Dataframe.isnull().values.ravel().sum()
            if(dropped > (0.05*len(Files))):
                raise Exception ("Too many columns were dropped from the dataframe", num_dropped)
            return Dataframe

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
    #Read the data inputs from a JSON file
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", help="JSON file")
    args = parser.parse_args()
    file_path = args.json
    with open(file_path, 'r') as f:
        files_dict = json.load(f, strict=False)

    # Creating a list of files and file handles that can be used to process the files
    A_Files = files_dict['A_Files'].split("?")
    A_File_Handle = [pd.read_csv(A_Files[i], header=0) for i in range(len(A_Files))]
    NAG_Files = files_dict['NAG_Files'].split("?")
    NAG_File_Handle = [pd.read_csv(NAG_Files[i]) for i in range(len(NAG_Files))]
    WBC_Files = files_dict['WBC_Files'].split("?")
    WBC_File_Handle = [pd.read_csv(WBC_Files[i]) for i in range(len(WBC_Files))]
    CD45D_Files = files_dict['CD45D_Files'].split("?")
    CD45D_File_Handle = [pd.read_csv(CD45D_Files[i]) for i in range(len(CD45D_Files))]
    CD45L_Files = files_dict['CD45L_Files'].split("?")
    CD45L_File_Handle = [pd.read_csv(CD45L_Files[i]) for i in range(len(CD45L_Files))]
    CD19CD10C_Files = files_dict['CD19CD10C_Files'].split("?")
    CD19CD10C_File_Handle = [pd.read_csv(CD19CD10C_Files[i]) for i in range(len(CD19CD10C_Files))]
    CD34_Files = files_dict['CD34_Files'].split("?")
    CD34_File_Handle = [pd.read_csv(CD34_Files[i]) for i in range(len(CD34_Files))]
    CD19PL_Files = files_dict['CD19PL_Files'].split("?")
    CD19PL_File_Handle = [pd.read_csv(CD19PL_Files[i]) for i in range(len(CD19PL_Files))]
    CD19NL_Files = files_dict['CD19NL_Files'].split("?")
    CD19NL_File_Handle = [pd.read_csv(CD19NL_Files[i]) for i in range(len(CD19NL_Files))]
    KPB_Files = files_dict['KPB_Files'].split("?")
    KPB_File_Handle = [pd.read_csv(KPB_Files[i]) for i in range(len(KPB_Files))]
    LPB_Files = files_dict['LPB_Files'].split("?")
    LPB_File_Handle = [pd.read_csv(LPB_Files[i]) for i in range(len(LPB_Files))]
    CD3CD16T_Files = files_dict['CD3CD16T_Files'].split("?")
    CD3CD16T_File_Handle = [pd.read_csv(CD3CD16T_Files[i]) for i in range(len(CD3CD16T_Files))]
    NK_Files = files_dict['NK_Files'].split("?")
    NK_File_Handle = [pd.read_csv(NK_Files[i]) for i in range(len(NK_Files))]
    NBNT_Files = files_dict['NBNT_Files'].split("?")
    NBNT_File_Handle = [pd.read_csv(NBNT_Files[i]) for i in range(len(NBNT_Files))]
    T_Files = files_dict['T_Files'].split("?")
    T_File_Handle = [pd.read_csv(T_Files[i]) for i in range(len(T_Files))]
    Metrics_Path = files_dict['Metrics_Path']
    Models_Path = files_dict['Models_Path']

    files = [A_File_Handle, NAG_File_Handle, WBC_File_Handle, CD45D_File_Handle, CD45L_File_Handle,
    CD19CD10C_File_Handle, CD34_File_Handle, CD19PL_File_Handle, CD19NL_File_Handle, KPB_File_Handle,
    LPB_File_Handle, CD3CD16T_File_Handle,NK_File_Handle, NBNT_File_Handle, T_File_Handle]


    #Create a list of all possible cell type classifications
    classes = ["A", "NAG", "ANAG", "WBC", "NAGWBC", "CD45D", "CD45L", "WBCCD45D", "WBCCD45DL", "CD19CD10C", "CD34", "CD45DCD19CD10C",
     "CD45DCD34", "CD19PL", "CD19NL", "CD45LCD19PL", "CD45LCD19NL", "KPB", "LPB",
    "CD19PLKPB", "CD19PLLPB", "CD3CD16T", "NK", "NBNT", "T", "CD19NLCD3CD16T", "CD19NLNK", "CD19NLT", "CD19NLNBNT"]
    val = 0

    #The below code is running the subset classification code on each subsequent file in order to clearly establish the file hierarchy and classification diagram
    A = pd.DataFrame()
    NAG = pd.DataFrame()
    A = Preprocessing(A_Files, A_File_Handle, A)
    open("results.txt", "w").close()
    add_to_file("results.txt", classes, A, val)

    NAG = Preprocessing(NAG_Files, NAG_File_Handle, NAG)
    val += 1
    add_to_file("results.txt", classes, NAG, val)

    ANAG = pd.DataFrame()
    ANAG = Subset_Setup(A, NAG, ANAG)
    val+=1
    add_to_file("results.txt", classes, ANAG, val)

    open("sanity_checks.txt", "w").close()
    sanity_check(ANAG,NAG)

    Basic_Classification(ANAG, "A_NAG.txt", Metrics_Path, Models_Path)


    WBC = pd.DataFrame()
    WBC = Preprocessing(WBC_Files, WBC_File_Handle, WBC)
    val+=1
    add_to_file("results.txt", classes, WBC, val)
    NAGWBC = pd.DataFrame()
    NAGWBC = Subset_Setup(NAG, WBC, NAGWBC)
    val+=1
    add_to_file("results.txt", classes, NAGWBC, val)
    sanity_check(NAGWBC, WBC)
    Basic_Classification(NAGWBC, "NAG_WBC.txt", Metrics_Path, Models_Path)


    CD45D = pd.DataFrame()
    CD45L = pd.DataFrame()
    CD45D = Preprocessing(CD45D_Files, CD45D_File_Handle, CD45D)
    CD45L = Preprocessing(CD45L_Files, CD45L_File_Handle, CD45L)
    val+=1
    add_to_file("results.txt", classes, CD45D, val)
    val+=1
    add_to_file("results.txt", classes, CD45L, val)
    WBCCD45D = pd.DataFrame()
    WBCCD45L = pd.DataFrame()
    WBCCD45D = Subset_Setup(WBC, CD45D, WBCCD45D)
    WBCCD45L = Subset_Setup(WBC, CD45L, WBCCD45L)

    sanity_check(WBCCD45D, CD45D)
    sanity_check(WBCCD45L, CD45L)

    Basic_Classification(WBCCD45D, "WBC_CD45D.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, WBCCD45D, val)
    Basic_Classification(WBCCD45L, "WBC_CD45L.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, WBCCD45L, val)


    CD19CD10C = pd.DataFrame()
    CD34 = pd.DataFrame()
    CD19CD10C = Preprocessing(CD19CD10C_Files, CD19CD10C_File_Handle, CD19CD10C)
    CD34 = Preprocessing(CD34_Files, CD34_File_Handle, CD34)
    val+=1
    add_to_file("results.txt", classes, CD19CD10C, val)
    val+=1
    add_to_file("results.txt", classes, CD34, val)
    CD45DCD19CD10C = pd.DataFrame()
    CD45DCD34 = pd.DataFrame()
    CD45DCD19CD10C = Subset_Setup(CD45D, CD19CD10C, CD45DCD19CD10C)
    CD45DCD34 = Subset_Setup(CD45D, CD34, CD45DCD34)
    sanity_check(CD45DCD19CD10C,CD19CD10C)
    sanity_check(CD45DCD34, CD34)

    Basic_Classification(CD45DCD19CD10C, "CD45D_CD19CD10C.txt", Metrics_Path, Models_Path)
    Basic_Classification(CD45DCD34, "CD45D_CD34.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD45DCD19CD10C, val)
    val += 1
    add_to_file("results.txt", classes, CD45DCD34, val)


    CD19PL = pd.DataFrame()
    CD19NL = pd.DataFrame()
    CD19PL = Preprocessing(CD19PL_Files, CD19PL_File_Handle, CD19PL)
    CD19NL = Preprocessing(CD19NL_Files, CD19NL_File_Handle, CD19NL)
    val+=1
    add_to_file("results.txt", classes, CD19PL, val)
    val+=1
    add_to_file("results.txt", classes, CD19NL, val)
    CD45LCD19PL = pd.DataFrame()
    CD45LCD19NL = pd.DataFrame()
    CD45LCD19PL = Subset_Setup(CD45L, CD19PL, CD45LCD19PL)
    CD45LCD19NL = Subset_Setup(CD45L, CD19NL, CD45LCD19NL)
    sanity_check(CD45LCD19PL,CD19PL)
    sanity_check(CD45LCD19NL, CD19NL)

    Basic_Classification(CD45LCD19PL, "CD45L_CD19PL.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD45LCD19PL, val)
    Basic_Classification(CD45LCD19NL, "CD45L_CD19NL.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD45LCD19NL, val)

    KPB = pd.DataFrame()
    LPB = pd.DataFrame()
    KPB = Preprocessing(KPB_Files, KPB_File_Handle, KPB)
    LPB = Preprocessing(LPB_Files, LPB_File_Handle, LPB)
    val+=1
    add_to_file("results.txt", classes, KPB, val)
    val+=1
    add_to_file("results.txt", classes, LPB, val)
    CD19PLKPB = pd.DataFrame()
    CD19PLLPB = pd.DataFrame()
    CD19PLKPB = Subset_Setup(CD19PL, KPB, CD19PLKPB)
    CD19PLLPB = Subset_Setup(CD19PL, LPB, CD19PLLPB)
    sanity_check(CD19PLKPB,KPB)
    sanity_check(CD19PLLPB, LPB)

    Basic_Classification(CD19PLKPB, "CD19PL_KPB.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD19PLKPB, val)
    Basic_Classification(CD19PLLPB, "CD19PL_LPB.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD19PLLPB, val)

    CD3CD16T = pd.DataFrame()
    NK = pd.DataFrame()
    NBNT = pd.DataFrame()
    T = pd.DataFrame()
    CD3CD16T = Preprocessing(CD3CD16T_Files, CD3CD16T_File_Handle, CD3CD16T)
    NK = Preprocessing(NK_Files, NK_File_Handle, NK)
    NBNT = Preprocessing(NBNT_Files, NBNT_File_Handle, NBNT)
    T = Preprocessing(T_Files, T_File_Handle, T)
    val+=1
    add_to_file("results.txt", classes, CD3CD16T, val)
    val+=1
    add_to_file("results.txt", classes, NK, val)
    val+=1
    add_to_file("results.txt", classes, NBNT, val)
    val+=1
    add_to_file("results.txt", classes, T, val)
    CD19NLCD3CD16T = pd.DataFrame()
    CD19NLNK = pd.DataFrame()
    CD19NLNBNT = pd.DataFrame()
    CD19NLT = pd.DataFrame()
    CD19NLCD3CD16T = Subset_Setup(CD19NL, CD3CD16T, CD19NLCD3CD16T)
    CD19NLNK = Subset_Setup(CD19NL, NK, CD19NLNK)
    CD19NLNBNT = Subset_Setup(CD19NL, NBNT, CD19NLNBNT)
    CD19NLT = Subset_Setup(CD19NL, T, CD19NLT)
    sanity_check(CD19NLCD3CD16T,CD3CD16T)
    sanity_check(CD19NLNK, NK)
    sanity_check(CD19NLNBNT, NBNT)
    sanity_check(CD19NLT, T)


    Basic_Classification(CD19NLCD3CD16T, "CD19NL_CD3CD16T.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD19NLCD3CD16T, val)
    Basic_Classification(CD19NLNK, "CD19NL_NK.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD19NLNK, val)
    Basic_Classification(CD19NLT, "CD19NL_T.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD19NLT, val)
    Basic_Classification(CD19NLNBNT, "CD19NL_NBNT.txt", Metrics_Path, Models_Path)
    val += 1
    add_to_file("results.txt", classes, CD19NLNBNT, val)


if __debug__:
    print('Debug ON')
else:
    print('Debug OFF')

if __name__== "__main__":
  main()
