# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os.path
from sklearn.metrics import plot_confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

import h5py
import json
from imblearn.under_sampling import RandomUnderSampler



parser = argparse.ArgumentParser()


parser.add_argument("-d", "--descriptor", help="Descriptor, puede ser gray color edge hog decaf \n default gray")
parser.add_argument("-mdir", "--memedirectory", help="Directorio con imagenes de memes default .\\data\\MemesDataSet\\Meme")
parser.add_argument("-nmdir", "--nomemedirectory", help="Directorio con imagenes de no memes default .\\data\\MemesDataSet\\No-meme")
parser.add_argument("-s", "--sticker", help="Cargar stickers, poner flag para usar stickers en las clases", action="store_true")
parser.add_argument("-cv", "--crossvalidation", help="Iteraciones para validación cruzada default 3",default=3, type=int)
args = parser.parse_args()

computer = ""
svcparams = {}
knnparams = {}
mlpparams = {}

if args.descriptor == "gray":
    descriptorName = "Gray Histogram"
    svcparams = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
    knnparams = {'algorithm': 'ball_tree', 'n_neighbors': 20, 'p': 2}
    mlpparams = {'activation': 'tanh', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    #0.47 sec
elif args.descriptor == "color":
    descriptorName = "Color Layout"
    svcparams = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
    knnparams = {'algorithm': 'kd_tree', 'n_neighbors': 3, 'p': 7}
    mlpparams = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': [230, 16], 'learning_rate': 'adaptive', 'solver': 'adam'}
    #0.0033 sec
elif args.descriptor == "edge":
    descriptorName = "Edge Histogram"
    svcparams = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
    knnparams = {'algorithm': 'ball_tree', 'n_neighbors': 20, 'p': 2}
    mlpparams = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'adam'}
    #0.021 sec
elif args.descriptor == "hog":
    descriptorName = "Histogram of Oriented Gradients"
    svcparams = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
    knnparams = {'algorithm': 'ball_tree', 'n_neighbors': 12, 'p': 7}
    mlpparams = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'adam'}
    #0.025 sec
elif args.descriptor == "decaf":
    descriptorName = "DeCAF7"
    svcparams = {'C': 14, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
    knnparams =  {'algorithm': 'kd_tree', 'n_neighbors': 12, 'p': 2}
    mlpparams = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (1000, 300), 'learning_rate': 'adaptive', 'solver': 'sgd'}
    #0.8 sec
elif args.descriptor == "resnet":
    descriptorName = "ResNet152"
    svcparams = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
    knnparams = {'algorithm': 'ball_tree', 'n_neighbors': 20, 'p': 2}
    mlpparams = {'activation': 'tanh', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'sgd'}
else:
    print("No se especificó descriptor, utilizando histograma de gris por defecto")
    descriptorName = "Gray Histogram"
    args.descriptor = "gray"
    svcparams = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
    knnparams = {'algorithm': 'ball_tree', 'n_neighbors': 20, 'p': 2}
    mlpparams = {'activation': 'tanh', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'sgd'}
   
memedir = ""
nomemedir = ""





if args.memedirectory == None:
    memedir = ".\\data\\MemesDataSet\\Meme"
else:
    memedir = args.memedirectory
    
if args.nomemedirectory == None:
    nomemedir = ".\\data\\MemesDataSet\\No-meme"
else:
    nomemedir = args.nomemedirectory
    



memeFeature = []
memeFname = []
memeFile = args.descriptor + "-meme.h5"

print("Cargando descriptores memes guardados "+ memeFile)
with h5py.File(memeFile, 'r') as hf:
    memeFeature = hf["features"][:]
    memeFname = hf["filename"][:]
    


noMemeFeature = []
noMemeFname = []
noMemeFile = args.descriptor + "-no-meme.h5"

print("Cargando descriptores no memes guardados "+ noMemeFile)
with h5py.File(noMemeFile, 'r') as hf:
    noMemeFeature = hf["features"][:]
    noMemeFname = hf["filename"][:]
    
    
    
    
stickerFeature = []
stickerFname = []
stickerFile = args.descriptor + "-sticker.h5"
print("Cargando descriptores stickers guardados "+ stickerFile)
with h5py.File(stickerFile, 'r') as hf:
    stickerFeature = hf["features"][:]
    stickerFname = hf["filename"][:]
    
    


labelMeme = np.ones((memeFeature.shape[0],1))
labelNoMeme = np.zeros((noMemeFeature.shape[0],1))
labelSticker = np.full((stickerFeature.shape[0],1),2)

features = []
labels= []

features = np.concatenate((noMemeFeature, memeFeature,stickerFeature))
labels = np.concatenate((labelNoMeme, labelMeme,labelSticker))


features = np.squeeze(features)
labels = np.squeeze(labels)

undersampler = RandomUnderSampler(random_state=0)
features, labels = undersampler.fit_resample(features, labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 0)

#train_features =features
#test_features = features
#train_labels = labels
#test_labels = labels
scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_features)
train_features = scaling.transform(train_features)
test_features = scaling.transform(test_features)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
 

kf = StratifiedKFold(n_splits = args.crossvalidation) 
classifiers = [RandomForestClassifier(n_estimators=300)]
classifiers.append(GaussianNB())
classifiers.append(LogisticRegression(max_iter=2000, solver="sag"))
classifiers.append(MLPClassifier(**mlpparams))
classifiers.append(SVC(**svcparams))
classifiers.append(KNeighborsClassifier(**knnparams))

plotdata = []
plottestdata = []
metrics_data = open('.\métricas\\'  + args.descriptor + '\\' + args.descriptor + '.txt','w')
for clf in classifiers:    

    metrics = {'acc': [], 'f1': [], 'precision': [], 'recall': []}
    
    for train_index, test_index in kf.split(train_features, train_labels):
        Xtrain, Xtest = train_features[train_index], train_features[test_index]
        Ytrain, Ytest = train_labels[train_index], train_labels[test_index]
        clf.fit(Xtrain,Ytrain)
        clfpredictions = clf.predict(Xtest) 
        clfP = precision_score(Ytest,clfpredictions, average="macro")
        clfR = recall_score(Ytest,clfpredictions, average="macro")
        clfA = accuracy_score(Ytest,clfpredictions)
        clfF = f1_score(Ytest,clfpredictions, average="macro")
        metrics["acc"].append(clfA)
        metrics["f1"].append(clfF)
        metrics["precision"].append(clfP)
        metrics["recall"].append(clfR)
    
    print("---")
    print(clf)
    acc_data = "acc: " + str(np.round(np.array(metrics['acc']).mean(), decimals=4)) + " ± " + str(np.round(2*np.array(metrics['acc']).std(), decimals=4))
    precision_data = "precision: " + str(np.round(np.array(metrics['precision']).mean(), decimals=4))+  " ± " + str(np.round(2*np.array(metrics['precision']).std(),decimals=4))
    recall_data = "recall: " + str(np.round(np.array(metrics['recall']).mean(), decimals=4))+ " ± "+  str(np.round(2*np.array(metrics['recall']).std(),decimals=4))
    f1_data = "f1: " + str(np.round(np.array(metrics['f1']).mean(), decimals=4))+ " ± "+ str(np.round(2*np.array(metrics['f1']).std(), decimals=4))
    print(acc_data)
    print(precision_data)
    print(recall_data)
    print(f1_data)
    print("test data")
    metrics_data.write('---\n')
    metrics_data.write(str(clf).split('(')[0] +'\n')
    metrics_data.write(acc_data +'\n')
    metrics_data.write(precision_data +'\n')
    metrics_data.write(recall_data +'\n')
    metrics_data.write(f1_data +'\n')
    
    clf.fit(train_features,train_labels)
    clfpredictions = clf.predict(test_features) 
    clfP = precision_score(test_labels,clfpredictions, average="macro")
    clfR = recall_score(test_labels,clfpredictions, average="macro")
    clfA = accuracy_score(test_labels,clfpredictions)
    clfF = f1_score(test_labels,clfpredictions, average="macro")
    print("accuracy",np.round(clfA,decimals=4),"precision",np.round(clfP,decimals=4),"recall",np.round(clfR,decimals=4),"f1", np.round(clfF,decimals=4))
    
    plotdata.append(np.array([
    [np.round(np.array(metrics['acc']).mean(), decimals=4), np.round(2*np.array(metrics['acc']).std(), decimals=4)],
    [np.round(np.array(metrics['precision']).mean(), decimals=4), np.round(2*np.array(metrics['precision']).std(),decimals=4)],
    [np.round(np.array(metrics['recall']).mean(), decimals=4), np.round(2*np.array(metrics['recall']).std(),decimals=4)],
    [np.round(np.array(metrics['f1']).mean(), decimals=4), np.round(2*np.array(metrics['f1']).std(), decimals=4)]]))
    plottestdata.append(np.array([clfP,clfR,clfA,clfF]))

    plot_confusion_matrix(clf,test_features, test_labels, display_labels=['No Meme', 'Meme', 'Sticker'],cmap="Blues", normalize='true')
    figname = '.\gráficos\\'  + args.descriptor + '\confusion matrix\\' + str(clf).split('(')[0] + '-norm.png'
    plt.title('Matriz de confusión para descriptor '  + args.descriptor + '\n con clasificador ' + str(clf).split('(')[0] + ' normalizada')

    plt.savefig(figname)
    plot_confusion_matrix(clf,test_features, test_labels, display_labels=['No Meme', 'Meme', 'Sticker'],cmap="Blues")
    figname = '.\gráficos\\'  + args.descriptor + '\confusion matrix\\' + str(clf).split('(')[0] + '.png'
    plt.title('Matriz de confusión para descriptor '  + args.descriptor + '\n con clasificador ' + str(clf).split('(')[0])

    plt.savefig(figname)
#    plt.show()
    
metrics_data.flush()
metrics_data.close()
plotdata=np.stack(plotdata) 
plottestdata=np.stack(plottestdata)
np.save(args.descriptor + "-plot-data.npy", plotdata)
np.save(args.descriptor + "-plot-test-data.npy", plottestdata)
    

