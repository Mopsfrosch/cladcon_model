#!/bin/bash
from Model import ClassificationModel
import tensorflow as tf
import argparse
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.utils import class_weight, compute_sample_weight
import pandas as pd
import numpy as ngp
import seaborn as sns
import matplotlib.pyplot as plt
import tracemalloc
from keras_balanced_batch_generator import make_generator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization


def get_encoded_labels(labels):
    maxval = np.max(labels) + 1
    return np.eye(maxval)[labels]

def get_binary_labels(y_train):
    labels = np.where(y_train == 14, 0, y_train)
    labels = np.where(y_train != 0, 0, 1)
    return labels

def clean_accession(row):
    row = str(row['Accession']).split('/')[-1]
    row = row.split('.txt')[0]
    return row

def main(args,filter_,window,dense,k,mode,filtered=0,featurenum=100,nEst=10):
    accuracies = []
    balancedaccuracies = []
    senspecs = []
    cfms = []
    traintimes = []
    predtimes = []
    trainmems = []
    predmems = []
    for fold in range(5):
        statusfile = pd.read_csv('meta/statusfile.csv')
        statusfile = statusfile[['Accession','age','sex']]
        sex_bin = np.where(statusfile['sex'] == 'female',1,0)
        statusfile['sex'] = sex_bin
        statusfile = statusfile.set_index('Accession')
        statusfile['age'] = statusfile['age'].apply(pd.to_numeric, errors='coerce')
        statusfile['age'] = statusfile['age'].fillna(statusfile['age'].median())
        if args.comb==0:
            col_names = pd.read_csv('preprocessed_full/'+str(fold)+'/feature_names.txt', sep="\t").iloc[:,0].tolist()
            X_train = pd.read_csv('preprocessed_full/'+str(fold)+'/train/feature_vectors.csv', names=col_names, header=None)
            y_train = pd.read_csv('preprocessed_full/'+str(fold)+'/train/labels.txt', names=['label']).values.ravel()
            X_test = pd.read_csv('preprocessed_full/'+str(fold)+'/test/feature_vectors.csv', names=col_names, header=None)
            y_test = pd.read_csv('preprocessed_full/'+str(fold)+'/test/labels.txt', names=['label']).values.ravel()
        if args.comb==1:
            col_names = pd.read_csv('preprocessed_comb/'+str(fold)+'/feature_names.txt', sep="\t").iloc[:,0].tolist()
            X_train = pd.read_csv('preprocessed_comb/'+str(fold)+'/train/feature_vectors.csv', names=col_names, header=None)
            y_train = pd.read_csv('preprocessed_comb/'+str(fold)+'/train/labels.txt', names=['label']).values.ravel()
            X_test = pd.read_csv('preprocessed_comb/'+str(fold)+'/test/feature_vectors.csv', names=col_names, header=None)
            y_test = pd.read_csv('preprocessed_comb/'+str(fold)+'/test/labels.txt', names=['label']).values.ravel()
        if args.comb==2:
            col_names = pd.read_csv('preprocessed/0/'+str(fold)+'/feature_names.txt', sep="\t").iloc[:,0].tolist()
            X_train = pd.read_csv('preprocessed/0/'+str(fold)+'/train/feature_vectors.csv', names=col_names, header=None)
            y_train = pd.read_csv('preprocessed/0/'+str(fold)+'/train/labels.txt', names=['label']).values.ravel()
            X_test = pd.read_csv('preprocessed/0/'+str(fold)+'/test/feature_vectors.csv', names=col_names, header=None)
            y_test = pd.read_csv('preprocessed/0/'+str(fold)+'/test/labels.txt', names=['label']).values.ravel()


        if args.filtered ==1:
            print('WE USE RFE WITH RANDOM FOREST')
            model_tree = RandomForestClassifier(random_state=args.seed, n_estimators=nEst, max_depth=20)
            sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=featurenum, step=1, verbose=0)
            X_train = sel_rfe_tree.fit_transform(X_train, y_train)
            X_test = sel_rfe_tree.transform(X_test)
        else:
            print('NO FEATURE SELECTION')
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()   


        if args.binary == 1:
            y_train = get_binary_labels(y_train)
            y_test = get_binary_labels(y_test)
            model_tree = RandomForestClassifier(random_state=args.seed, n_estimators=nEst, max_depth=20)
            sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=featurenum, step=1, verbose=0)
            X_train = sel_rfe_tree.fit_transform(X_train, y_train)
            X_test = sel_rfe_tree.transform(X_test)
            y_train = get_encoded_labels(y_train)
            y_test = get_encoded_labels(y_test)

        else:
            y_train = get_encoded_labels(y_train)
            y_test = get_encoded_labels(y_test)
        classnum = len(y_train[1])

        width = len(X_train[0])
        height = 1
        X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1],1)
        BATCH_SIZE = args.batch
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE,drop_remainder=True)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE,drop_remainder=True)
        

        model = ClassificationModel(args,width, height, train_dataset, test_dataset, filter_, window, classnum=classnum, loaded=False,dense=dense,seed=args.seed,mode=mode,filtered=filtered, featurenum = featurenum,fold=fold)

        accuracies.append(model.accuracy)
        balancedaccuracies.append(model.balancedAccuracy)
        senspecs.append(model.senspec)
        cfms.append(model.cfm)
        traintimes.append(model.trainingstime)
        predtimes.append(model.predictiontime)
        trainmems.append(model.trainingmemory)
        predmems.append(model.predictionmemory)

    print('all accuracies: ',accuracies)
    print('avg accuracy: ',np.array(accuracies).mean())
    print('std accuracies: ',np.array(accuracies).std())
    print('all balanced accuracies: ',balancedaccuracies)
    print('avg balanced accuracy: ',np.array(balancedaccuracies).mean())
    print('std balanced accuracies: ',np.array(balancedaccuracies).std())

    sensitivity = np.zeros((5,14))
    specificity = np.zeros((5,14))
    for j, ss in enumerate(senspecs):
        for i in range(14):
            sensitivity[j][i]=senspecs[j][i]['sensitivity']
            specificity[j][i]=senspecs[j][i]['specificity']
    sensitivity = np.mean(sensitivity,axis=0)
    specificity = np.mean(specificity,axis=0)
    print('avg sensitivities: ')
    print(sensitivity)
    print('---------------------------------------')
    print('avg specificities:')
    print(specificity)

    print('avg sensitivities: ', file=text_file)
    print(sensitivity, file=text_file)
    print('---------------------------------------', file=text_file)
    print('avg specificities:', file=text_file)
    print(specificity, file=text_file)

    aggr = []
    for i,conf in enumerate(cfms):
        aggr.append((cfms[i].T/cfms[i].sum(axis=1)).T)
    oben = np.zeros((14,14))
    for i,a in enumerate(aggr):
        oben = oben + aggr[i]
    result = oben/5

    result = np.around(result,decimals=2)
    print(result)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'model settings')
    parser.add_argument('--filter',type=int, default=128, help='number of filters') 
    parser.add_argument('--window',type=int, default=3, help='window size') 
    parser.add_argument('--dense',type=int, default=512, help='size of dense layer') 
    parser.add_argument('--dense1',type=int, default=512, help='size of dense layer') 
    parser.add_argument('--dense2',type=int, default=512, help='size of dense layer') 
    parser.add_argument('--dense3',type=int, default=512, help='size of dense layer') 
    parser.add_argument('--meta',type=int, default=5, help='size of dense layer') 
    parser.add_argument('--k',type=int, default=5, help='k fold cross validation')
    parser.add_argument('--filtered',type=int, default=0, help='1 == filtered data or 0 == use full data set ')
    parser.add_argument('--epochs',type=int, default=50, help='1 == filtered data or 0 == use full data set ')
    parser.add_argument('--data',type=float,default=1.0, help='percentage of data used for training')
    parser.add_argument('--featurenum',type=int,default=100, help='number of selected features')
    parser.add_argument('--batch',type=int,default=32, help='batch size')
    parser.add_argument('--nEst',type=int,default=100, help='number of estimators')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--binary',type=int,default=0)
    parser.add_argument('--task',type=int,default='tissue', help='tissue, comb, full')
    args = parser.parse_args()

    st = time.time()

    main(args,args.filter,args.window,args.dense,args.k,args.mode,args.filtered,args.featurenum)
    et = time.time()
    elapsed_time = et - st

    print('Execution time full:', elapsed_time, 'seconds')
    print('Execution time full:', (elapsed_time / 60), 'minutes')
    print('Execution time full:', (elapsed_time / 3600), 'hours')
    print('all args: ',args)
    print('done')