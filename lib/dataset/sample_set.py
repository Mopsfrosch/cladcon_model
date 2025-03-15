'''Adapted from https://github.com/pfnet-research/head_model/tree/master'''


import os
import shutil

import numpy as np
import pandas as pd
from lib.util import util

import xgboost as xgb
from sklearn.feature_selection import SelectFromModel


class SampleSet(object):

    def __init__(self, feature_vectors, instance_names, labels, feature_names=None):
        self.feature_vectors = feature_vectors
        self.instance_names = instance_names
        self.labels = labels
        self.feature_names = feature_names
        self.validate()

    def validate(self):
        N, D = self.feature_vectors.shape
        util.assert_equal(len(self.instance_names), N)
        util.assert_equal(len(self.labels), N)

    def __len__(self):
        return len(self.feature_vectors)

    def equal(self, other, ignore_fv):
        if ignore_fv:
            return (np.array_equal(self.instance_names, other.instance_names)
                    and np.array_equal(self.labels, other.labels))
        else:
            return (np.allclose(self.feature_vectors, other.feature_vectors)
                    and np.array_equal(self.instance_names,
                                       other.instance_names)
                    and np.array_equal(self.labels, other.labels))

    def __eq__(self, other):
        return self.equal(other, False)

    def __add__(self, other):
        feature_vectors = np.vstack(
            (self.feature_vectors, other.feature_vectors))
        instance_names = np.hstack(
            (self.instance_names, other.instance_names))
        labels = np.hstack((self.labels, other.labels))
        return SampleSet(feature_vectors, instance_names, labels)

    def split(self, train_idx, test_idx):
        # N = len(self)
        all_idx = np.hstack((train_idx, test_idx))
        all_idx.sort()
        # print('all_idx: ',all_idx)
        # print('len self: ',N)
        # print('len all idx: ', len(all_idx))
        # print('shape feature vectors in sample set: ',self.feature_vectors.shape)
        # #if not np.array_equal(np.arange(N), all_idx):
        # #    raise ValueError('train_idx and test_idx are invalid')
        # X_train = self.feature_vectors[train_idx]
        # X_test = self.feature_vectors[test_idx]
        # #y_train = self.labels[train_idx]
        # if False:
        #     print('NO DATA LEAK FROM FEATURE SELECTION')
        #     doit = True
        #     #for i in range(1):
        #     # if doit:
        #     #     xgbclf = xgb.XGBClassifier(random_state=0,importance_type='total_cover')
        #     #     xgbclf.fit(X_train,y_train)
        #     #     scores = xgbclf.feature_importances_ #Return type Dict[str, float | List[float]]
        #     #     print('--------------SCORES-------------------')
        #     #     print(scores)
        #     #     doit = False
        #     #clf = xgb.XGBClassifier()
        #     #clf.fit(X_train,y_train)
        #     X_train_orig = pd.DataFrame(X_train_orig,columns=self.feature_names)
        #     X_test_df = pd.DataFrame(X_test,columns=self.feature_names)
        #     X_test_control = X_test_df[['hsa-miR-2861', 'hsa-miR-149-3p', 'hsa-miR-4463']].to_numpy()
        #     X_test_df = X_test_df.drop(['hsa-miR-2861', 'hsa-miR-149-3p', 'hsa-miR-4463'], axis=1)
        #     X_train_control = X_train_orig[['hsa-miR-2861', 'hsa-miR-149-3p', 'hsa-miR-4463']].to_numpy()
        #     X_train_orig = X_train_orig.drop(['hsa-miR-2861', 'hsa-miR-149-3p', 'hsa-miR-4463'], axis=1)
        #     selection = SelectFromModel(estimator=xgb.XGBClassifier(n_estimators=1,max_depth=1), max_features=100).fit(X_train_orig,y_train)
        #     #selection = SelectFromModel(estimator=xgbclf, prefit=True, max_features=100).fit(X_train,y_train)
        #     self.feature_names = np.concatenate((selection.get_feature_names_out(), np.array(['hsa-miR-2861', 'hsa-miR-149-3p', 'hsa-miR-4463'])),axis=0)
        #     print('len features nach selection: ',len(self.feature_names))
        #     print(self.feature_names)
        #     X_train = np.concatenate((selection.transform(X_train_orig),X_train_control),axis=1)
            
        #     X_test = np.concatenate((selection.transform(X_test_df),X_test_control),axis=1)
        #     print('X_train shape: ',X_train.shape)
        #     # if doit:
        #     #     xgbclf = xgb.XGBClassifier(random_state=0,importance_type='total_cover')
        #     #     xgbclf.fit(X_train_orig,y_train)
        #     #     scores = xgbclf.feature_importances_ #Return type Dict[str, float | List[float]]
        #     #     print('--------------SCORES-------------------')
        #     #     print(scores)
        #     #     doit = False

        feature_train, names_train, labels_train = self.feature_vectors[train_idx], self.instance_names[train_idx], self.labels[train_idx]
        feature_train, names_train, labels_train = util.shuffle(feature_train, names_train, labels_train)
        feature_test, names_test, labels_test = self.feature_vectors[test_idx], self.instance_names[test_idx], self.labels[test_idx]
        feature_test, names_test, labels_test = util.shuffle(feature_test, names_test, labels_test)

        train = SampleSet(feature_train,
                          names_train,
                          labels_train)
        test = SampleSet(feature_test,
                         names_test,
                         labels_test)
        return train, test, self.feature_names

    def sort(self):
        idx = np.argsort(self.instance_names)
        self.feature_vectors = self.feature_vectors[idx]
        self.instance_names = self.instance_names[idx]
        self.labels = self.labels[idx]

    @staticmethod
    def load(in_dir):
        feature_vectors = np.loadtxt(
            os.path.join(in_dir, 'feature_vectors.csv'),
            np.float32, delimiter=',', ndmin=2)
        instance_names = util.load_column(
            os.path.join(in_dir, 'instance_names.txt'))
        labels = util.load_column(
            os.path.join(in_dir, 'labels.txt'), np.int32)
        return SampleSet(feature_vectors, instance_names, labels)

    def dump(self, out_dir):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

        np.savetxt(os.path.join(out_dir, 'feature_vectors.csv'),
                   self.feature_vectors, delimiter=',')

        with open(os.path.join(out_dir, 'instance_names.txt'), 'w+') as o:
            o.write('\n'.join(self.instance_names))

        with open(os.path.join(out_dir, 'labels.txt'), 'w+') as o:
            o.write('\n'.join(map(str, self.labels)))
