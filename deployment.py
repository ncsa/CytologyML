import pandas as pd
import numpy as np
import json
import os
import argparse
import logging, sys
import scikitplot as skplt
import matplotlib.pyplot as plt
import pickle
from training import Preprocessing

from sklearn import tree, metrics
from sklearn.externals.joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sklearn.externals.joblib.numpy_pickle


def evaluation_matrix(name, predicted, known, Metrics_Path):
    '''This function is converting dataframes to sets so that arithmetic can be performed to calculate TP, FP, FN on the sets'''
    if(known.empty):
        f = open(Metrics_Path+"evaluation_matrix.txt", "a")
        f.write(name + " is empty\n")
        f.write("------------")
        print(name + " is empty\n")
        return
    elif(predicted.empty):
        f = open(Metrics_Path+"evaluation_matrix.txt", "a")
        f.write(name + " is predicted to be empty by the model\n")
        f.write("------------")
        print(name + " is predicted to be empty by the model\n")
        return

    f = open(Metrics_Path+"evaluation_matrix.txt", "a")

    predicted_list = (predicted.values).tolist()
    predicted_set = set(map(tuple, predicted_list))

    known = known.drop(['Time'], axis=1)
    known_list = (known.values).tolist()
    known_set = set(map(tuple,known_list))

    f.write(name + ":" + '\n')
    #Use set intersection methods
    f.write("Length of Known: " + str(len(known_set)) + '\n')
    f.write("Length of Predicted: " + str(len(predicted_set)) + '\n')

    #True positives should be in both the predicted and known sets
    TP = predicted_set.intersection(known_set)
    f.write("NUMBER OF TP: " + str(len(TP)) + '\n')

    #False positives will be in the predicted set but not in the known set
    FP = predicted_set - known_set
    f.write("NUMBER OF FP: " + str(len(FP)) + '\n')

    #False negatives will be in the known set but not the predicted set
    FN = known_set - predicted_set
    f.write("NUMBER OF FN: " + str(len(FN)) + '\n')

    f.write("------------" + '\n')

def method_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="Method to use")
    parser.add_argument("-j", "--json", help="Path to JSON file")
    args = parser.parse_args()
    print( "Method Being Used: {} ".format(
        args.method,
        ))
    vals = [args.method, args.json]
    return vals

def Check_Attributes_Dataframe(Dataframe):
    """Check the attributes of the dataframe to check that everything is included."""
    print("############################")
    print(Dataframe.shape)
    print(Dataframe.columns)
    print(Dataframe.head())
    print("############################")


def main():
    study = 0
    index = 0
    vals = method_input()
    file_path = vals[1]
    method = vals[0]
    A = pd.DataFrame()
    NAG = pd.DataFrame()
    ANAG = pd.DataFrame()
    WBC = pd.DataFrame()
    CD45D = pd.DataFrame()
    CD45L = pd.DataFrame()
    CD19CD10C = pd.DataFrame()
    CD34 = pd.DataFrame()
    CD19PL = pd.DataFrame()
    CD19NL = pd.DataFrame()
    KPB = pd.DataFrame()
    LPB = pd.DataFrame()
    CD3CD16T = pd.DataFrame()
    NK = pd.DataFrame()
    NBNT = pd.DataFrame()
    T = pd.DataFrame()
    Popstats = []

    #Read the data inputs from the JSON file. If debugging mode is on, read from the JSON sent to training. Otherwise, read from a smaller set of files in deployment_files.json
    if(__debug__):
        with open(file_path, 'r') as f:
            files_dict = json.load(f, strict=False)

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
        Checks_Path = files_dict['Checks_Path']
        Study_Name = files_dict['Study_Name']

        A = Preprocessing(A_Files, A_File_Handle, A)
        NAG = Preprocessing(NAG_Files, NAG_File_Handle, NAG)
        WBC = Preprocessing(WBC_Files, WBC_File_Handle, WBC)
        CD45D = Preprocessing(CD45D_Files, CD45D_File_Handle, CD45D)
        CD45L = Preprocessing(CD45L_Files, CD45L_File_Handle, CD45L)
        CD19CD10C = Preprocessing(CD19CD10C_Files, CD19CD10C_File_Handle, CD19CD10C)
        CD34 = Preprocessing(CD34_Files, CD34_File_Handle, CD34)
        CD19PL = Preprocessing(CD19PL_Files, CD19PL_File_Handle, CD19PL)
        CD19NL = Preprocessing(CD19NL_Files, CD19NL_File_Handle, CD19NL)
        KPB = Preprocessing(KPB_Files, KPB_File_Handle, KPB)
        LPB = Preprocessing(LPB_Files, LPB_File_Handle, LPB)
        CD3CD16T = Preprocessing(CD3CD16T_Files, CD3CD16T_File_Handle, CD3CD16T)
        NK = Preprocessing(NK_Files, NK_File_Handle, NK)
        NBNT = Preprocessing(NBNT_Files, NBNT_File_Handle, NBNT)
        T = Preprocessing(T_Files, T_File_Handle, T)

    else:
        with open(file_path, 'r') as f:
            files_dict = json.load(f, strict=False)
        A_Files = files_dict['A_Files'].split("?")
        A_File_Handle = [pd.read_csv(A_Files[i], header=0) for i in range(len(A_Files))]
        Metrics_Path = files_dict['Metrics_Path']
        Models_Path = files_dict['Models_Path']
        Checks_Path = files_dict['Checks_Path']
        A = Preprocessing(A_Files, A_File_Handle, A)

    with open(Metrics_Path+"evaluation_matrix.txt", 'w') as a:
        a.write("Study Name: " + Study_Name + '\n')
        a.close()
    A = A.drop(['Time'], axis=1)
    Popstats.append(A.shape[0])
    A_NAG_DT = load(Models_Path+"A_NAG_DT.pkl", 'r')
    A_NAG_LR = load(Models_Path+"A_NAG_LR.pkl", 'r')
    A_NAG_NB = load(Models_Path+"A_NAG_NB.pkl", 'r')
    A = A.loc[:, A.columns != 'Type']
    y_DT = A_NAG_DT.predict(A)
    y_DT_prob = A_NAG_DT.predict_proba(A)
    y_LR = A_NAG_LR.predict(A)
    y_NB = A_NAG_NB.predict(A)
    options = [y_DT, y_LR, y_NB]
    list = ["DT", "LR", "NB"]
    y = options[list.index(method)]

    A['Type'] = y
    NAG_predicted = A.loc[A['Type'] == 1]
    evaluation_matrix('NAG', NAG_predicted, NAG, Metrics_Path)
    logging.debug("NAG\t" + str(NAG_predicted.shape[0]))
    Popstats.append(NAG_predicted.shape[0])
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
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    NAG_predicted['Type'] = y
    WBC_predicted = NAG_predicted.loc[NAG_predicted['Type'] == 1]
    evaluation_matrix('WBC', WBC_predicted, WBC, Metrics_Path)
    logging.debug("WBC\t" + str(WBC_predicted.shape[0]))
    Popstats.append(WBC_predicted.shape[0])

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
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    WBC_predicted['Type'] = y
    CD45D_predicted = WBC_predicted.loc[WBC_predicted['Type'] == 1]
    evaluation_matrix('CD45D', CD45D_predicted, CD45D, Metrics_Path)
    logging.debug("CD45D\t" + str(CD45D_predicted.shape[0]))
    Popstats.append(CD45D_predicted.shape[0])

    WBC_predicted = WBC_predicted.drop(['Type'], axis=1)
    CD45L_predicted = pd.DataFrame()
    WBC_CD45L_DT = load(Models_Path+"WBC_CD45L_DT.pkl", 'r')
    WBC_CD45L_LR = load(Models_Path+"WBC_CD45L_LR.pkl", 'r')
    WBC_CD45L_NB = load(Models_Path+"WBC_CD45L_NB.pkl", 'r')
    y_DT = WBC_CD45L_DT.predict(WBC_predicted)
    y_LR = WBC_CD45L_LR.predict(WBC_predicted)
    y_NB = WBC_CD45L_NB.predict(WBC_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    WBC_predicted['Type'] = y
    CD45L_predicted = WBC_predicted.loc[WBC_predicted['Type'] == 1]
    evaluation_matrix('CD45L', CD45L_predicted, CD45L, Metrics_Path)
    logging.debug("CD45L\t" + str(CD45L_predicted.shape[0]))
    Popstats.append(CD45L_predicted.shape[0])

    ################################
    ################################
    ################################
    ################################

    CD45D_predicted = CD45D_predicted.drop(['Type'], axis=1)
    CD19CD10C_predicted = pd.DataFrame()
    CD45D_CD19CD10C_DT = load(Models_Path+"CD45D_CD19CD10C_DT.pkl", 'r')
    CD45D_CD19CD10C_LR = load(Models_Path+"CD45D_CD19CD10C_LR.pkl", 'r')
    CD45D_CD19CD10C_NB = load(Models_Path+"CD45D_CD19CD10C_NB.pkl", 'r')
    y_DT = CD45D_CD19CD10C_DT.predict(CD45D_predicted)
    y_LR = CD45D_CD19CD10C_LR.predict(CD45D_predicted)
    y_NB = CD45D_CD19CD10C_NB.predict(CD45D_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD45D_predicted['Type'] = y
    CD19CD10C_predicted = CD45D_predicted.loc[CD45D_predicted['Type'] == 1]
    print("DTYPES --------")
    print(CD19CD10C_predicted.dtypes)
    evaluation_matrix('CD19CD10C', CD19CD10C_predicted, CD19CD10C, Metrics_Path)
    logging.debug("CD19CD10C\t" + str(CD19CD10C_predicted.shape[0]))
    Popstats.append(CD19CD10C_predicted.shape[0])


    CD45D_predicted = CD45D_predicted.drop(['Type'], axis=1)
    CD34_predicted = pd.DataFrame()
    CD45D_CD34_DT = load(Models_Path+"CD45D_CD34_DT.pkl", 'r')
    CD45D_CD34_LR = load(Models_Path+"CD45D_CD34_LR.pkl", 'r')
    CD45D_CD34_NB = load(Models_Path+"CD45D_CD34_NB.pkl", 'r')
    y_DT = CD45D_CD34_DT.predict(CD45D_predicted)
    y_LR = CD45D_CD34_LR.predict(CD45D_predicted)
    y_NB = CD45D_CD34_NB.predict(CD45D_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD45D_predicted['Type'] = y
    CD34_predicted = CD45D_predicted.loc[CD45D_predicted['Type'] == 1]
    evaluation_matrix('CD34', CD34_predicted, CD34, Metrics_Path)
    logging.debug("CD34\t" + str(CD34_predicted.shape[0]))
    Popstats.append(CD34_predicted.shape[0])

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
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD45L_predicted['Type'] = y
    CD19PL_predicted = CD45L_predicted.loc[CD45L_predicted['Type'] == 1]
    evaluation_matrix('CD19PL', CD19PL_predicted, CD19PL, Metrics_Path)
    logging.debug("CD19PL\t" + str(CD19PL_predicted.shape[0]))
    Popstats.append(CD19PL_predicted.shape[0])


    CD45L_predicted = CD45L_predicted.drop(['Type'], axis=1)
    CD19NL_predicted = pd.DataFrame()
    CD45L_CD19NL_DT = load(Models_Path+"CD45L_CD19NL_DT.pkl", 'r')
    CD45L_CD19NL_LR = load(Models_Path+"CD45L_CD19NL_LR.pkl", 'r')
    CD45L_CD19NL_NB = load(Models_Path+"CD45L_CD19NL_NB.pkl", 'r')
    y_DT = CD45L_CD19NL_DT.predict(CD45L_predicted)
    y_LR = CD45L_CD19NL_LR.predict(CD45L_predicted)
    y_NB = CD45L_CD19NL_NB.predict(CD45L_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD45L_predicted['Type'] = y
    CD19NL_predicted = CD45L_predicted.loc[CD45L_predicted['Type'] == 1]
    evaluation_matrix('CD19NL', CD19NL_predicted, CD19NL, Metrics_Path)
    logging.debug("CD19NL\t" + str(CD19NL_predicted.shape[0]))
    Popstats.append(CD19NL_predicted.shape[0])

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
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD19PL_predicted['Type'] = y
    KPB_predicted = CD19PL_predicted.loc[CD19PL_predicted['Type'] == 1]
    evaluation_matrix('KPB', KPB_predicted, KPB, Metrics_Path)
    logging.debug("KPB\t" + str(KPB_predicted.shape[0]))
    Popstats.append(KPB_predicted.shape[0])

    CD19PL_predicted = CD19PL_predicted.drop(['Type'], axis=1)
    LPB_predicted = pd.DataFrame()
    CD19PL_LPB_DT = load(Models_Path+"CD19PL_LPB_DT.pkl", 'r')
    CD19PL_LPB_LR = load(Models_Path+"CD19PL_LPB_LR.pkl", 'r')
    CD19PL_LPB_NB = load(Models_Path+"CD19PL_LPB_NB.pkl", 'r')
    y_DT = CD19PL_LPB_DT.predict(CD19PL_predicted)
    y_LR = CD19PL_LPB_LR.predict(CD19PL_predicted)
    y_NB = CD19PL_LPB_NB.predict(CD19PL_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD19PL_predicted['Type'] = y
    LPB_predicted = CD19PL_predicted.loc[CD19PL_predicted['Type'] == 1]
    evaluation_matrix('LPB', LPB_predicted, LPB, Metrics_Path)
    logging.debug("LPB\t" + str(LPB_predicted.shape[0]))
    Popstats.append(LPB_predicted.shape[0])

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
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD19NL_predicted['Type'] = y
    CD3CD16T_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
    evaluation_matrix('CD3CD16T', CD3CD16T_predicted, CD3CD16T, Metrics_Path)
    logging.debug("CD3CD16T\t" + str(CD3CD16T_predicted.shape[0]))
    Popstats.append(CD3CD16T_predicted.shape[0])

    CD19NL_predicted = CD19NL_predicted.drop(['Type'], axis=1)
    NK_predicted = pd.DataFrame()
    CD19NL_NK_DT = load(Models_Path+"CD19NL_NK_DT.pkl", 'r')
    CD19NL_NK_LR = load(Models_Path+"CD19NL_NK_LR.pkl", 'r')
    CD19NL_NK_NB = load(Models_Path+"CD19NL_NK_NB.pkl", 'r')
    y_DT = CD19NL_NK_DT.predict(CD19NL_predicted)
    y_LR = CD19NL_NK_LR.predict(CD19NL_predicted)
    y_NB = CD19NL_NK_NB.predict(CD19NL_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD19NL_predicted['Type'] = y
    NK_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
    evaluation_matrix('NK', NK_predicted, NK, Metrics_Path)
    logging.debug("NK\t" + str(NK_predicted.shape[0]))
    Popstats.append(NK_predicted.shape[0])

    CD19NL_predicted = CD19NL_predicted.drop(['Type'], axis=1)
    NBNT_predicted = pd.DataFrame()
    CD19NL_NBNT_DT = load(Models_Path+"CD19NL_NBNT_DT.pkl", 'r')
    CD19NL_NBNT_LR = load(Models_Path+"CD19NL_NBNT_LR.pkl", 'r')
    CD19NL_NBNT_NB = load(Models_Path+"CD19NL_NBNT_NB.pkl", 'r')
    y_DT = CD19NL_NBNT_DT.predict(CD19NL_predicted)
    y_LR = CD19NL_NBNT_LR.predict(CD19NL_predicted)
    y_NB = CD19NL_NBNT_NB.predict(CD19NL_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD19NL_predicted['Type'] = y
    NBNT_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
    evaluation_matrix('NBNT', NBNT_predicted, NBNT, Metrics_Path)
    logging.debug("NBNT\t" + str(NBNT_predicted.shape[0]))
    Popstats.append(NBNT_predicted.shape[0])

    CD19NL_predicted = CD19NL_predicted.drop(['Type'], axis=1)
    T_predicted = pd.DataFrame()
    CD19NL_T_DT = load(Models_Path+"CD19NL_T_DT.pkl", 'r')
    CD19NL_T_LR = load(Models_Path+"CD19NL_T_LR.pkl", 'r')
    CD19NL_T_NB = load(Models_Path+"CD19NL_T_NB.pkl", 'r')
    y_DT = CD19NL_T_DT.predict(CD19NL_predicted)
    y_LR = CD19NL_T_LR.predict(CD19NL_predicted)
    y_NB = CD19NL_T_NB.predict(CD19NL_predicted)
    options = [y_DT, y_LR, y_NB]
    y = options[list.index(method)]
    CD19NL_predicted['Type'] = y
    T_predicted = CD19NL_predicted.loc[CD19NL_predicted['Type'] == 1]
    evaluation_matrix('T', T_predicted, T, Metrics_Path)
    logging.debug("T\t" + str(T_predicted.shape[0]))
    Popstats.append(T_predicted.shape[0])

    ################################
    ################################
    print(Popstats)
if __name__== "__main__":
  main()
