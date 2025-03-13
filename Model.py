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

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


class ClassificationModel:

    def __init__(self, args, img_w, img_h,training_set,test_set,filter,window,
                 classnum,loaded, path='',folder='grayscale',modelname='all_',channelnumber=1,dense=4096,
                 dense0=4096,dense1=4096,dense2=4096,seed=0,mode=0,filtered=0,featurenum=100,fold=0):
        keras.utils.set_random_seed(seed)
        mode=1
        print('use mode ',mode)
        self.meta = False
        self.image_width = img_w
        self.image_height = img_h
        self.model = None
        self.loaded = loaded
        self.last_batch = False
        self.filter = args.filter
        self.window = args.window
        self.classnum = classnum
        self.path = path
        self.folder = 'output/'+folder
        self.modelname = modelname
        self.channel = channelnumber
        self.dense = args.dense
        self.dense1 = args.dense1
        self.dense2 = args.dense2
        self.dense3 = args.dense3
        if args.con2 ==1:
            self.con2 = True
        else:
            self.con2=False
        if args.con3==1:
            self.con3 = True
        else:
            self.con3=False
        self.meta_dense = args.meta_dense
        self.featurenum = featurenum
        print('feature num in model: ',featurenum)
        self.test_set = test_set
        self.training_set = training_set
        self.i = int(str(args.dense)+str(args.dense1)+str(args.dense2)+str(args.dense3)+str(args.filter)+str(args.window)+str(args.meta)+str(args.seed)+str(fold)+str(args.comb)+str(args.filtered))
        self.mode = mode
        self.filtered = args.filtered
        self.binary = args.binary
        self.fold = fold
        self.seed = args.seed
        self.comb = args.comb
        

        self.batch_size = 32
        
        self.epochs = args.epochs
        self.rate = 0.0001

        self.define_callbacks()
        st = time.time()
        tracemalloc.start()
        self.model = self.ensemble_model(training_set)
        et = time.time()
        self.trainingmemory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_time = et - st
        self.trainingstime = elapsed_time
        print('Execution time training:', elapsed_time, 'seconds')
        print('Execution time training:', (elapsed_time / 60), 'minutes')
        print('Execution time training:', (elapsed_time / 3600), 'hours')
        stp = time.time()
        tracemalloc.start()
        self.accuracy, self.balancedAccuracy, self.senspec, self.cfm = self.evaluate_model(test_set)
        etp = time.time()
        self.predictionmemory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_timep = etp - stp
        self.predictiontime = elapsed_timep
        print('traintime: ',self.trainingstime, 'testtime: ', self.predictiontime)
        print('Execution time prediction:', elapsed_timep, 'seconds')
        print('Execution time prediction:', (elapsed_timep / 60), 'minutes')
        print('Execution time prediction:', (elapsed_timep / 3600), 'hours')
  
    
    def ensemble_model(self, training_set):
        data_input = keras.Input(shape=(self.image_height, self.image_width, self.channel))
        kernel_size = (self.image_height,self.window)
        print('kernel size: ',kernel_size)
        print('img height, window: ',self.image_height,self.window)
    
        data_features = layers.Conv2D(self.filter, kernel_size, padding='same', activation='relu')(data_input)
        data_features = layers.BatchNormalization()(data_features)
        data_features = layers.Conv2D(self.filter, kernel_size, padding='same', activation='relu')(data_features)
        data_features = layers.BatchNormalization()(data_features)
        data_features = layers.MaxPooling2D(padding='same')(data_features)
        data_features = layers.Dropout(.2, input_shape=(2,))(data_features)
        if self.con2:
            data_features = layers.Conv2D(self.filter, kernel_size, padding='same', activation='relu')(data_features)
            data_features = layers.BatchNormalization()(data_features)
            data_features = layers.Conv2D(self.filter, kernel_size, padding='same', activation='relu')(data_features)
            data_features = layers.BatchNormalization()(data_features)
            data_features = layers.MaxPooling2D(padding='same')(data_features)
            data_features = layers.Dropout(.3, input_shape=(2,))(data_features)
        if self.con3:
            data_features = layers.Conv2D(self.filter, kernel_size, padding='same', activation='relu')(data_features)
            data_features = layers.BatchNormalization()(data_features)
            data_features = layers.Conv2D(self.filter, kernel_size, padding='same', activation='relu')(data_features)
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
        x = np.squeeze(data.copy())
        with open('_df_data.txt', 'w') as outfile:
            for slice_2d in x:
                np.savetxt(outfile, slice_2d)
        print('x data shape in train: ',data.shape)
        print('y data shape in train: ',y_train.shape)
      
        y_weight = np.concatenate(np.array([np.argmax(y, axis=1) for x,y in training_set]))
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(y_weight),
                                                          y=y_weight)
        
        class_weights = dict(enumerate(class_weights))
       
        print('data: ',data.shape)
        print('y_train: ',y_train.shape)
        history = model.fit(x=data,y=y_train, epochs=self.epochs, validation_split=0.1, shuffle=False, class_weight=class_weights,
                                  callbacks=self.define_callbacks(),verbose=1)
        
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['balanced_accuracy'])
        plt.plot(history.history['val_balanced_accuracy'])
        plt.title('model balanced_accuracy')
        plt.ylabel('balanced_accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('train_balanced_accuracy_plot.png')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('train_loss_plot.png')

        return model


    def define_callbacks(self):
        early_stopping = EarlyStopping(patience=8, min_delta=0.001, restore_best_weights=True)

        checkpoint_cb = ModelCheckpoint(self.path+"model/"+str(self.i)+"_classifier_trained.h5", save_best_only=True)
        #early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
        logger_cb = CSVLogger(self.path+'model/logs/'+str(self.i)+'_classifier_trained.csv', separator="|")
        return [checkpoint_cb,# early_stopping_cb,
                logger_cb,early_stopping]

    def train_model(self, training_set, validation_set):
        #y_train = np.concatenate([y for x, y in training_set], axis=0)
        y_train = np.concatenate([np.argmax(y, axis=1) for x, y in training_set], axis=0)
        #y_train = np.argmax(y_train, axis=1)  # Convert one-hot to index
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(y_train),
                                                          y=y_train)
        class_weights = dict(enumerate(class_weights))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.CategoricalAccuracy()],run_eagerly = True)
        history = self.model.fit(training_set, epochs=self.epochs, validation_data=validation_set, shuffle=True, class_weight=class_weights,
                                 callbacks=self.define_callbacks())
        #self.model.save(self.path+'model/'+str(self.i)+'_classifier_trained.h5')
        return history
    
    
    def plot_history(self):
        data_frame = pandas.DataFrame(self.history.history)
        data_frame.plot(figsize=(7, 3))
        plt.xlabel('Epochs')
        plt.ylabel('Sparse categorical cross-entropy')
        plt.savefig(self.folder+'/'+str(self.i)+'_history.png')


    def plot_metric(self, metric):
        train_metrics = self.history.history[metric]
        val_metrics = self.history.history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.savefig(self.folder+'/'+str(self.i)+'_plot.png')

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
            if self.comb == 1:
                resu.to_csv('fixed_results/comb_'+str(self.fold)+'_results.csv',index=None)
            if self.comb==0:
                resu.to_csv('fixed_results/full_'+str(self.fold)+'_results.csv',index=None)
            if self.comb==2:
                resu.to_csv('fixed_results/tissue_'+str(self.fold)+'_results.csv',index=None)
        #y_pred.to_csv(self.folder+'/'+str(self.i)+'_ally_pred.csv',index=False)
        print('Classification Report')
        #classes = [0,1]
        #classes = ['BC', 'BL', 'BT', 'CC', 'EC', 'GC', 'GL', 'HC', 'LK', 'OV', 'PC', 'PR', 'SA', 'no cancer']
        
        if self.binary:
            print(classification_report(Y_test, y_pred,target_names=['no cancer','cancer']))    
        else:
            print(classification_report(Y_test, y_pred))#,target_names=classes))
        #print('test type: ',type(Y_test),' pred type: ',type(y_pred))
        Y_test_ = pd.DataFrame(Y_test)
        #Y_test_.to_csv('Y_test.csv')
        Y_pred_ = pd.DataFrame(y_pred)
        #Y_pred_.to_csv('Y_pred.csv')
        balancedAccuracy = balanced_accuracy_score(Y_test,y_pred)
        print('Balanced Accuracy: ', balancedAccuracy)
        print('Confusion Matrix')
        print(confusion_matrix(Y_test, y_pred))
        cfm = confusion_matrix(Y_test,y_pred)
        cfm_df = pd.DataFrame(cfm)
      #  cfm_df.to_csv(self.folder+'/'+str(self.i)+'_cfm.csv', index=False, header=False)


        cmd = ConfusionMatrixDisplay.from_predictions(
            Y_test, y_pred
        )
        cmd.figure_.savefig(self.folder+'/'+str(self.i)+'_confusion_matrix.png')

        # specificity
        print('test',type(Y_test))
        print('pred',y_pred.shape)
        report = self.sensitivity_and_specificity(Y_test, y_pred)
        print(report)
        report = pd.DataFrame(report)
        senspec = report
        report.to_csv(self.folder+'/'+str(self.i)+'_report.csv')
        try:
            tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        except:
            print('co confusion matrix')
        #print('tn: ',tn,' fp: ',fp,' fn: ',fn,' tp: ',tp)

        #test_loss, f1_score, test_acc, test_precision, test_recall, FN, FP, TN, TP, CategoricalAccuracy = self.model.evaluate(x={'miRNA': miRNA_data, 'meta': meta_data},y={'cancertype': y_test})
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
        return test_acc,balancedAccuracy, senspec, cfm
