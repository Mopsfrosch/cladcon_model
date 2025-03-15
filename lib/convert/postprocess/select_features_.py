'''Adapted from https://github.com/pfnet-research/head_model/tree/master'''


import math

import pandas as pd
import numpy as np

from lib.dataset import dataset as D
from lib.dataset import metadata as M
from lib.dataset import sample_set as S
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier

def select_features(dataset):
    train = dataset.train
    test = dataset.test
    metadata = dataset.metadata
    split = dataset.split
    random_state = dataset.random_state

    df_train = pd.DataFrame(train.feature_vectors,
                            columns=metadata.feature_names)
    df_test = pd.DataFrame(test.feature_vectors,
                           columns=metadata.feature_names)
    #importancescores = []
    print('THIS IS SPLIT NR ',split)
    #for rs in range(5):
    #print('THIS IS RS ',rs)
    model_tree = RandomForestClassifier(random_state=random_state, n_estimators=100, max_depth=20)
    sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=100, step=1, verbose=0)
    sel_rfe_tree.fit(df_train,train.labels)
    #clf = xgb.XGBClassifier(random_state=rs).fit(df_train,train.labels)
    #print('nur die scores: ',clf.feature_importances_)
    feature_names = sel_rfe_tree.get_feature_names_out()
    df = pd.DataFrame(feature_names)
    pd.DataFrame(df).to_csv('fs/'+str(random_state)+'/'+str(split)+'/feature_list.csv', header=None)
    #importancescores.append(np.array(sel_rfe_tree.get_feature_names_out()))
    # for rs in range(5):
    #     #print(' importance score files: ',pd.read_csv('fs/'+str(split)+'/'+str(rs)+'_XGBoost_feature_importances.csv',header=None).values.tolist())
    #     scores = pd.read_csv('fs/'+str(split)+'/'+str(rs)+'_XGBoost_feature_importances.csv',header=None)
    #     scores = scores.iloc[:, 1].tolist()
    #     importancescores.append(scores)
    #importancescores = np.stack(importancescores, axis=-1)
    #importancescores = np.array(importancescores)
    #is_df = pd.DataFrame(np.transpose(importancescores),index=metadata.feature_names)
    #is_df.to_csv('fs/'+str(split)+'/XGBoost_all_feature_importances.csv', header=None)
    #importancescores = np.stack(importancescores, axis=-1)
    #importancescores = np.mean(importancescores, axis=1)
    #fi = list(zip(metadata.feature_names,importancescores))
    #fi.sort(reverse=True, key= lambda x: x[1])
    #pd.DataFrame(fi).to_csv('fs'+str(random_state)+'/'+str(split)+'/XGBoost_avg_feature_importances.csv',index=None, header=None)

    #fi = fi[:100]
    #fi = list(list(zip(*fi))[0])
    #print('fi: ',fi)
    #print('fi type ',type(fi))
    #print('df train columns: ',df_train.columns)

   # selection = SelectFromModel(estimator=xgb.XGBClassifier(random_state=0), max_features=100).fit(df_train,train.labels)
   # selected = selection.get_feature_names_out()
   # df_train = selection.transform(df_train)
   # df_test = selection.transform(df_test)
    df_train = sel_rfe_tree.transform(df_train)
    df_test = sel_rfe_tree.transform(df_test)
    train = S.SampleSet(df_train,
                        train.instance_names,
                        train.labels)
    test = S.SampleSet(df_test,
                       test.instance_names,
                       test.labels)
    metadata = M.Metadata(feature_names,
                          metadata.label_names)
    dataset = D.Dataset(metadata, train, test, split, random_state)
    return dataset