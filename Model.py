#!/bin/bash
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, applications, optimizers, losses, metrics
from tensorflow.keras.callbacks import *
import tensorflow.keras as keras
import tracemalloc
import pandas
import time
import matplotlib.pyplot as plt
import os
import math
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils import class_weight, compute_sample_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import ConfusionMatrixDisplay

def balanced_accuracy(Y_test,y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
        return balanced_accuracy_score(Y_test,y_pred)


class ClassificationModel:

    def __init__(self, args, img_w, img_h, training_set, test_set, classnum, fold=0):
        keras.utils.set_random_seed(args.seed)

        self.image_width = img_w
        self.image_height = img_h
        self.last_batch = False
        self.filter = args.filter
        self.window = args.window
        self.classnum = classnum
        self.dense = args.dense
        self.dense1 = args.dense1
        self.dense2 = args.dense2
        self.dense3 = args.dense3
        self.featurenum = args.featurenum
        self.test_set = test_set
        self.training_set = training_set
        self.i = str(args.dense)+str(args.dense1)+str(args.dense2)+str(args.dense3)+str(args.filter)+str(args.window)+str(args.seed)+str(fold)+str(args.task)+str(args.featureselect)
        self.filtered = args.featureselect
        self.binary = args.binary
        self.fold = fold
        self.task = args.task
        self.batch_size = 32
        self.epochs = args.epochs
        self.rate = 0.0001

        self.define_callbacks()
        self.model = self.build_model(training_set)
        self.accuracy, self.balancedAccuracy, self.senspec, self.cfm = self.evaluate_model(test_set)
  
    
    def build_model(self, training_set):
        data_input = keras.Input(shape=(self.image_height, self.image_width, 1))
        kernel_size = (self.image_height,self.window)
    
        data_features = layers.Conv2D(self.filter, kernel_size, padding='same', activation='relu')(data_input)
        data_features = layers.BatchNormalization()(data_features)
        data_features = layers.MaxPooling2D(padding='same')(data_features)
        data_features = layers.Dropout(.4, input_shape=(2,))(data_features)

        data_features = layers.Flatten()(data_features)
        data_features = layers.Dense(self.dense, activation='relu')(data_features)
        data_features = layers.BatchNormalization()(data_features)
        data_features = layers.Dropout(.2)(data_features)
        data_features = layers.Dense(self.dense1, activation='relu')(data_features)
        data_features = layers.BatchNormalization()(data_features)
        data_features = layers.Dropout(.3)(data_features)
        data_features = layers.Dense(self.dense2, activation='relu')(data_features)
        data_features = layers.BatchNormalization()(data_features)
        data_features = layers.Dropout(.4)(data_features)
        data_features = layers.Dense(self.dense3, activation='relu')(data_features)

        cancer = layers.Dense(self.classnum, activation='sigmoid', name='cancertype')(data_features)

        model = keras.Model(inputs=[data_input],outputs=[cancer])    
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[balanced_accuracy,'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.CategoricalAccuracy()],run_eagerly = True)

        y_train = np.vstack([np.array(y) for x,y in training_set])
        data = np.vstack([x for x,y in training_set])
      
        y_weight = np.concatenate(np.array([np.argmax(y, axis=1) for x,y in training_set]))
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(y_weight),
                                                          y=y_weight) 
        class_weights = dict(enumerate(class_weights))
       
        history = model.fit(x=data,y=y_train, epochs=self.epochs, validation_split=0.1, shuffle=False, class_weight=class_weights,
                                  callbacks=self.define_callbacks(),verbose=1)

        return model


    def define_callbacks(self):
        early_stopping = EarlyStopping(patience=8, min_delta=0.001, restore_best_weights=True)

        checkpoint_cb = ModelCheckpoint('model/'+str(self.i)+'_classifier_trained.h5', save_best_only=True)
        logger_cb = CSVLogger('model/logs/'+str(self.i)+'_classifier_trained.csv', separator="|")
        return [checkpoint_cb,
                logger_cb, early_stopping]
    


    def sensitivity_and_specificity(self,y_true, y_pred, label_names=None):
        y_true = y_true.to_numpy()
        y_pred = y_pred.to_numpy()
        report = {}
        labels = np.unique(np.hstack((y_true, y_pred)))
        for l in labels:
            pos = (y_true == l).sum()
            neg = (y_true != l).sum()
            tp = ((y_true == l) & (y_pred == l)).sum()
            tn = ((y_true != l) & (y_pred != l)).sum()

            sensitivity = 0. if pos == 0 else tp / pos
            specificity = 0. if neg == 0 else tn / neg
            key = l if label_names is None else label_names[l]
            report[key] = {'sensitivity': sensitivity,
                           'specificity': specificity,
                           'support': pos}
        return report


    def evaluate_model(self, test_set):
        y_test = np.vstack([np.array(y) for x,y in test_set])
    
        data = np.vstack([x for x,y in test_set])
        predictions = self.model.predict(data)
        predictions_=pd.DataFrame(predictions)
    
        Y_test = np.concatenate([np.argmax(y, axis=1) for x, y in test_set], axis=0)
        y_pred = np.argmax(predictions, axis=1)  # Convert one-hot to index
        Y_test=pd.DataFrame(Y_test)
        Y_test.columns = ['true']
        y_pred=pd.DataFrame(y_pred)
        y_pred.columns = ['pred']
        if True:
            resu = pd.concat([Y_test, y_pred], axis=1)
            if self.task=='comb':
                resu.to_csv('results/comb_'+str(self.fold)+'_results.csv',index=None)
            if self.task=='full':
                resu.to_csv('results/full_'+str(self.fold)+'_results.csv',index=None)
            if self.task=='tissue':
                resu.to_csv('results/tissue_'+str(self.fold)+'_results.csv',index=None)

        print('Classification Report')
       
        if self.binary==1:
            print(classification_report(Y_test, y_pred))    
        else:
            print(classification_report(Y_test, y_pred))

        Y_test_ = pd.DataFrame(Y_test)
        Y_pred_ = pd.DataFrame(y_pred)
        balancedAccuracy = balanced_accuracy_score(Y_test,y_pred)
        print('Balanced Accuracy: ', balancedAccuracy)
        print('Confusion Matrix')
        cfm = confusion_matrix(Y_test,y_pred)
        print(cfm)

        # cmd = ConfusionMatrixDisplay.from_predictions(
        #     Y_test, y_pred
        # )
        # cmd.figure_.savefig(str(self.task)+'_confusion_matrix.png')

        senspec = self.sensitivity_and_specificity(Y_test, y_pred)
        print(senspec)
        try:
            tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        except:
            print('co confusion matrix')

        test_loss, f1_score, test_acc, test_precision, test_recall, FN, FP, TN, TP, CategoricalAccuracy = self.model.evaluate(x=data,y={'cancertype': y_test})
        print('\nTest loss:', test_loss)
        print('\nTest f1_score:', f1_score)
        print('\nTest accuracy:', test_acc)
        print('\nTest precision:', test_precision)
        print('\nTest recall:', test_recall)
        print('\nTest FN:', FN)
        print('\nTest FP:', FP)
        print('\nTest TN:', TN)
        print('\nTest TP:', TP)
        print('\nTest CategoricalAccuracy:', CategoricalAccuracy)
        res = [test_loss, test_acc, test_precision, test_recall, FN, FP, TN, TP, CategoricalAccuracy]#, tn, fp, fn, tp]
        return test_acc, balancedAccuracy, senspec, cfm
