'''Adapted from https://github.com/pfnet-research/head_model/tree/master'''


import glob
import os
import shutil
import sys
import pandas as pd
import numpy as np
from sklearn import model_selection

from lib.dataset import dataset as D
from lib.dataset import metadata as M
from lib.dataset import sample_set as S
from lib.util import util


def _validate(metadata, sample_set):
    util.assert_equal(len(sample_set.feature_vectors.shape), 2)
    N, D = sample_set.feature_vectors.shape
    print('N: ',N)
    print('D: ',D)
    util.assert_equal(len(metadata.feature_names), D)
    util.assert_equal(len(sample_set.instance_names), N)
    util.assert_equal(len(sample_set.labels), N)

    M = len(metadata.label_names)
    assert all([l < M for l in sample_set.labels]), 'invalid labels'


def dump_k_fold(out_dir, num_fold,
                feature_names, label_names,
                feature_vectors, instance_names, labels,
                filters,random_state):

    """Dumps k fold cross validation dataset

    Expected output
    out/
    ├── 0
    │   ├── feature_names.txt
    │   ├── label_names.txt
    │   ├── test
    │   │   ├── feature_vectors.csv
    │   │   ├── instance_names
    │   │   └── labels.txt
    │   └── train
    │       ├── feature_vectors.csv
    │       ├── instance_names
    │       └── labels.txt
    .
    .
    .
    └── 4
        ├── feature_names.txt
        ├── label_names.txt
        ├── test
        │   ├── feature_vectors.csv
        │   ├── instance_names
        │   └── labels.txt
        └── train
            ├── feature_vectors.csv
            ├── instance_names
            └── labels.txt
    """

    if os.path.exists(out_dir):
        print('Directory %s exists. Remove it.' % out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    metadata = M.Metadata(feature_names, label_names)
    sample_set = S.SampleSet(feature_vectors, instance_names, labels, feature_names)
    #print('sample set: ',sample_set)
    #print('instance names: ',instance_names)
    _validate(metadata, sample_set)
    
    print('all filters: ',filters)

    for percentage in [1.0]:#,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
        skf = model_selection.StratifiedKFold(num_fold,random_state=random_state,shuffle=True)
        place_holder = np.zeros_like(labels[:, None])    
        if percentage !=1.0:
            place_holder_, _, labels_, _ = model_selection.train_test_split(place_holder, labels, train_size=float(percentage), stratify=labels)
        else:
            place_holder_ =place_holder
            labels_ = labels
        # train, test, _ = sample_set.split(np.array(list(range(len(instance_names)-1))), np.array([len(instance_names)-1]))
        # all_data = D.Dataset(metadata, train, test)
        # train, test = all_data.train.feature_vectors, all_data.test.feature_vectors
        # for f in filters:
        #     all_data = f(all_data)
        # out_dir_for_this_fold = os.path.join(out_dir, 'ALLEN')#,str(percentage))
        
        # all_data.dump(out_dir_for_this_fold)
        for i, (train_idx, test_idx) in enumerate(skf.split(place_holder_, labels_)):
            train, test, feature_names = sample_set.split(train_idx, test_idx)
            print('train: ',type(train_idx))
            print('train: ',train_idx)
            metadata = M.Metadata(feature_names, label_names)
            print('dum k fold - feature_names ',len(feature_names),type(feature_names))
            print('dum k fold - label_names ',len(label_names))
            print('train_idx, test_idx ',train_idx, test_idx)
            print('iteration: ',i)
            dataset = D.Dataset(metadata, train, test, i, random_state)
            for f in filters:
                dataset = f(dataset)
            out_dir_for_this_fold = os.path.join(out_dir, str(i))#,str(percentage))
            dataset.dump(out_dir_for_this_fold)

    if False:
        for j in range(5):
            train_idx = []
            test_idx = []
            testa = pd.read_csv('/home/hanna/head_model/preprocess/'+str(j)+'_test_acc.csv', header=None).values.tolist()
            testa = [item for sublist in testa for item in sublist]
            traina = pd.read_csv('/home/hanna/head_model/preprocess/'+str(j)+'_train_acc.csv', header=None).values.tolist()
            traina = [item for sublist in traina for item in sublist]
            print('instance names: ', instance_names)
            both = []
            for idx, i in enumerate(instance_names):
                i = i.split('/')[-1]
                i = i.split('.')[0]
                if i in testa:
                    test_idx.append(idx)
                    both.append(i)
                if i in traina:
                    both.append(i)
                    train_idx.append(idx)

            for name in instance_names:
                name = name.split('/')[-1]
                name = name.split('.')[0]
                if name not in both:
                    print(name)
            print('train len: ', len(train_idx))
            print('test len: ', len(test_idx))
            print('both: ', len(train_idx + test_idx))
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            train, test = sample_set.split(train_idx, test_idx)
            dataset = D.Dataset(metadata, train, test)
            for f in filters:
                dataset = f(dataset)
            out_dir_for_this_fold = os.path.join(out_dir, str(j))
            dataset.dump(out_dir_for_this_fold)



    dump_myself(sys.argv[0], os.path.join(out_dir, 'script'))


def dump_myself(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        print('Directory %s exists. Remove it.' % dst_dir)
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    for f in glob.glob(os.path.join(src_dir, '**/*.py'), recursive=True):
        rel_path = os.path.relpath(f, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        dir_name = os.path.dirname(dst_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        shutil.copy2(f, dst_path)


'''
i=0
train_idx = []
test_idx = []
testa = pd.read_csv('/home/hanna/head_model/preprocess/test_acc.csv', header=None).values.tolist()
testa = [item for sublist in testa for item in sublist]
traina = pd.read_csv('/home/hanna/head_model/preprocess/train_acc.csv', header=None).values.tolist()
traina = [item for sublist in traina for item in sublist]
print('instance names: ', instance_names)
both = []
for idx, i in enumerate(instance_names):
    i = i.split('/')[-1]
    i = i.split('.')[0]
    if i in testa:
        test_idx.append(idx)
        both.append(i)
    if i in traina:
        both.append(i)
        train_idx.append(idx)
i = 0

for name in instance_names:
    name = name.split('/')[-1]
    name = name.split('.')[0]
    if name not in both:
        print(name)
print('train len: ', len(train_idx))
print('test len: ', len(test_idx))
print('both: ', len(train_idx+test_idx))
train_idx = np.asarray(train_idx)
test_idx = np.asarray(test_idx)
# print(train_idx)
# print(type(train_idx))
# print(test_idx)
# print(type(test_idx))

# print(train_idx)
# print('typetrain: ',type(train_idx))
# print(test_idx)
# print('typetest: ',type(test_idx))''' 