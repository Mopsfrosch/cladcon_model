#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import pandas as pd

from lib.convert import postprocess
from lib.convert import preprocess
from lib.io import dump
from lib.io import load
from lib.io import sample_directory
from lib.util import util
from lib.util.id_converter import IDConverter
from lib.util.feature_converter import convert_feature_vectors
import tracemalloc, time



def fetch(root_dir, dataset, filters):
    feature_vectors = []
    instance_names = []
    labels = []

    feature_id_converter = IDConverter()
    label_id_converter = IDConverter()

    for dir_name, label_name in dir_names:
        dir_name = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_name):
            print(dir_name + ' not found.')
            continue

        print('Processing ' + dir_name)

        feature_vectors_, instance_names_ = load.fetch_dir(
            dir_name, feature_id_converter, filters)
        #print('feat: ',feature_vectors_)
        #print('inst: ', instance_names_)
        lid = label_id_converter.to_id(label_name)
        labels_ = [lid] * len(feature_vectors_)

        util.assert_equal(len(feature_vectors_), len(instance_names_))
        util.assert_equal(len(feature_vectors_), len(labels_))

        feature_vectors += feature_vectors_
        instance_names += instance_names_
        labels += labels_

    # Convert data
    feature_names = feature_id_converter.id2name
    label_names = label_id_converter.id2name
    feature_vectors = convert_feature_vectors(
        feature_vectors, feature_id_converter.unique_num, True)
    instance_names = np.array(instance_names)
    labels = np.array(labels)

    # shuffle
    #feature_vectors, instance_names, labels = util.shuffle(
    #    feature_vectors, instance_names, labels)

    return feature_names, label_names, feature_vectors, instance_names, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessor')
    parser.add_argument('--in-dir', '-i', type=str, default='.')
    parser.add_argument('--out-dir', '-o', type=str, default='out')
    parser.add_argument('--num-fold', '-k', type=int, default=5)
    parser.add_argument('--dataset', '-d', type=str, default='comb', #full,our
                        choices=('full', 'our', 'test_ternary_large',
                                 'test_binary_large', 'test_binary_small','comb'))
    parser.add_argument('--use-ngs-collated-mirna-only', '-u',
                        action='store_true')
    parser.add_argument('--whiten-before-split', '-w', action='store_true',
                        help='Normalize the expression levels '
                        'along feature dimensions. '
                        'the Note that it whitens BEFORE splitting '
                        'the dataset into train and test.')
    parser.add_argument('--whiten-after-split', '-W', action='store_true',
                        help='Normalize the expression levels '
                        'along feature dimensions. '
                        'Note that it whitens AFTER splitting '
                        'the dataset into train and test.')
    parser.add_argument('--normalize', '-n', action='store_true',
                        help='Normalize the expression levels '
                        'along samples (after splitting).')
    parser.add_argument('--use-important-mirna-only', '-I',
                        type=int, default=-1,
                        help='# of adopted miRNAs based on importance score. '
                        'If it is a non-positive value, '
                        'it does not filter miRNAs '
                        'based on importance scores.')
    parser.add_argument('--remove-unused-mirnas', '-D', action='store_true',
                        help='If true, we remove reported unused miRNAs '
                        'from feature dimensions.')
    parser.add_argument('--random_state', '-rs', type=int, default='0')
    args = parser.parse_args()
    st = time.time()
    tracemalloc.start()

    if args.dataset == 'full':
        dir_names = sample_directory.DIR_NAMES
    elif args.dataset == 'comb':
        print('use comb')
        dir_names = sample_directory.DIR_NAMES_comb
    elif args.dataset == 'our':
        print('use ours')
        dir_names = sample_directory.DIR_NAMES_OUR
    elif args.dataset == 'test_ternary_large':
        dir_names = sample_directory.TEST_DIR_NAMES_TERNARY_LARGE
    elif args.dataset == 'test_binary_large':
        dir_names = sample_directory.TEST_DIR_NAMES_BINARY_LARGE
    elif args.dataset == 'test_binary_small':
        dir_names = sample_directory.TEST_DIR_NAMES_BINARY_SMALL
    else:
        raise ValueError('invalid dataset type:{}'.format(args.dataset))
    print('args.normalize: ',args.normalize)
    # args.normalize = False
    print('all args: ',args)

    preprocess_filters = list(preprocess.DEFAULT_FILTERS)
    if args.remove_unused_mirnas:
        preprocess_filters.append(
            preprocess.remove_unused_mirna)
    if args.normalize:
        # We must keep negative controls because the normalize
        # operation needs them.
        # After the normalizer performs normalization,
        # it drops feature dimensions other than hsa miRNAs.
        preprocess_filters.remove(preprocess.use_hsa_mirna_only)
    if args.use_ngs_collated_mirna_only:
        preprocess_filters.append(
            preprocess.use_ngs_collated_mirna)
    if args.whiten_before_split:
        preprocess_filters.append(preprocess.whiten)

    

    d = fetch(args.in_dir, dir_names, preprocess_filters)
    #print('d: ',d)
    feature_names = d[0]
    print('feature names: ',type(feature_names),len(feature_names), feature_names[0])
    label_names = d[1]
    feature_vectors = d[2]
    print('feature vectors: ',type(feature_vectors), feature_vectors.shape, feature_vectors[0])
    instance_names = d[3]
    print('instance_names: ',type(instance_names),instance_names.shape)
    labels = d[4]
    print('types: ',type(feature_names),type(label_names),type(feature_vectors),type(labels))

    

    # data_df = pd.DataFrame(feature_vectors, columns=feature_names)
    # data_df.insert(0, 'instance_names', instance_names)
    # data_df.insert(1, 'labels', labels)
    # data_df.to_csv('raw_HEAD_data.csv')

    # data = pd.read_csv('input_all_HEAD_vsn_nonlog.csv')
    # data = data.drop(['NA'], axis=1)
    # data = data.T
    # new_header = data.iloc[0] #grab the first row for the header
    # data = data[1:] #take the data less the header row
    # data.columns = new_header #set the header row as the df header    
    # #labels = raw['labels'].to_numpy()# values.tolist()
    # #instance_names = raw['ID_REF'].to_numpy() #values.tolist()
    # #raw = raw.drop('labels',axis=1)
    # #raw = raw.drop('ID_REF',axis=1)
    # print('data: ',data)
    # feature_names = new_header#np.array(raw.columns.values.tolist())
    # print('feature vectors1: ',type(feature_vectors), feature_vectors.shape, feature_vectors[0])
    # feature_vectors = data.values.tolist()#match_order(instance_names,feature_names) #values.tolist()
    # print('feature vectors2: ',type(feature_vectors), feature_vectors.shape, feature_vectors[0])
    # print('labels: ',labels.shape)
    # print('instance_names: ',instance_names.shape)
    # print('feature_vectors: ',feature_vectors.shape)
    # print('label_names: ',len(label_names))
    # print('feature_names: ',feature_names)

    # data_df = pd.DataFrame(feature_vectors, columns=feature_names)
    # data_df.insert(0, 'instance_names', instance_names)
    # data_df.insert(1, 'labels', labels)
    # data_df.to_csv('raw_HEAD_data_nonlog.csv')

    # feature_vectors, instance_names, labels = util.shuffle(
    #     feature_vectors, instance_names, labels)

    postprocess_filters = []
    if args.whiten_after_split:
        print('postprocess_filters whiten')
        postprocess_filters.append(postprocess.whiten)
    if args.normalize:
        print('postprocess_filters normalize')
        postprocess_filters.append(postprocess.normalize)
    if args.use_important_mirna_only > 0:
        print('postprocess_filters importance score')
        postprocess_filters.append(
            # feature selection
            
            postprocess.create_importance_score_filter(
                args.use_important_mirna_only))
    if False:
        postprocess_filters.append(postprocess.select_features)

    dump.dump_k_fold(args.out_dir, args.num_fold,
                     feature_names, label_names,
                     feature_vectors, instance_names, labels,
                     postprocess_filters,args.random_state)
    print('done')
    et = time.time()
    elapsed_time = et - st
    print('full memory usage: ',tracemalloc.get_traced_memory())
    tracemalloc.stop()
    print('Execution time full:', elapsed_time, 'seconds')
    print('Execution time full:', (elapsed_time / 60), 'minutes')
    print('Execution time full:', (elapsed_time / 3600), 'hours')