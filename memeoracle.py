# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:17:52 2020

@author: albsa
"""

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


svcparams = {}
knnparams = {}
mlpparams = {}

svcparams["gray"] = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
knnparams["gray"] = {'algorithm': 'ball_tree', 'n_neighbors': 20, 'p': 2}
mlpparams["gray"] = {'activation': 'tanh', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'sgd'}

svcparams["color"] = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
knnparams["color"] = {'algorithm': 'kd_tree', 'n_neighbors': 3, 'p': 7}
mlpparams["color"] = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (500, 10), 'learning_rate': 'adaptive', 'solver': 'adam'}

svcparams["edge"] =  {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
knnparams["edge"] = {'algorithm': 'ball_tree', 'n_neighbors': 20, 'p': 2}
mlpparams["edge"] = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'adam'}

svcparams["hog"] = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
knnparams["hog"] = {'algorithm': 'ball_tree', 'n_neighbors': 12, 'p': 7}
mlpparams["hog"] = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'adam'}

svcparams["decaf"] = {'C': 14, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
knnparams["decaf"] =  {'algorithm': 'kd_tree', 'n_neighbors': 12, 'p': 2}
mlpparams["decaf"] = {'activation': 'relu', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (1000, 300), 'learning_rate': 'adaptive', 'solver': 'sgd'}

svcparams["resnet"] = {'C': 1, 'coef0': 0, 'degree': 1, 'kernel': 'rbf'}
knnparams["resnet"] = {'algorithm': 'ball_tree', 'n_neighbors': 20, 'p': 2}
mlpparams["resnet"] = {'activation': 'tanh', 'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': (3000), 'learning_rate': 'adaptive', 'solver': 'sgd'}


#"edge","hog","color","gray","decaf","resnet"
desc = ["edge","hog","color","gray","decaf","resnet"]
memeFeature = {}
noMemeFeature = {}
stickerFeature = {}
stickerFile = "-sticker2.h5"
noMemeFile = "-no-meme2.h5"
memeFile = "-meme2.h5"
fnameMeme = {}
fnameNoMeme = {}
fnameSticker = {}
for d in desc:
    for dct, ndct, f in ((memeFeature,fnameMeme,memeFile),(noMemeFeature,fnameNoMeme,noMemeFile),(stickerFeature,fnameSticker,stickerFile)):
        print("Cargando descriptores guardados "+ d+f)
        with h5py.File(d+f, 'r') as hf:
            dct[d] = hf["features"][:]
            ndct[d] = hf["filename"][:]
                
    




#shuffle data before undersampling
indicesMeme = np.arange(memeFeature[desc[0]].shape[0])
indicesNoMeme = np.arange(noMemeFeature[desc[0]].shape[0])
indicesSticker = np.arange(stickerFeature[desc[0]].shape[0])
np.random.shuffle(indicesMeme)
np.random.shuffle(indicesNoMeme)
np.random.shuffle(indicesSticker)

fnameMeme = np.asarray(fnameMeme[desc[0]],dtype="S")
fnameNoMeme = np.asarray(fnameNoMeme[desc[0]],dtype="S")
fnameSticker = np.asarray(fnameSticker[desc[0]],dtype="S")
fnameMeme = fnameMeme[indicesMeme]
fnameNoMeme = fnameNoMeme[indicesNoMeme]
fnameSticker = fnameSticker[indicesSticker]

for d in desc:
    memeFeature[d] = memeFeature[d][indicesMeme]
    noMemeFeature[d] = noMemeFeature[d][indicesNoMeme]
    stickerFeature[d] = stickerFeature[d][indicesSticker]

#undersample data
samples = len(memeFeature[desc[0]])
labelMeme = np.ones((samples,1))
labelNoMeme = np.zeros((samples,1))
labelSticker = np.full((samples,1),2)
fnameMeme = fnameMeme[:samples]
fnameNoMeme = fnameNoMeme[:samples]
fnameSticker = fnameSticker[:samples]

for d in desc:
    memeFeature[d] = memeFeature[d][:samples]
    noMemeFeature[d] = noMemeFeature[d][:samples]
    stickerFeature[d] = stickerFeature[d][:samples]
    
    
#shuffle and finish data
shuffleIndices = np.arange(samples*3)
np.random.shuffle(shuffleIndices)

features = {}
for d in desc:
    f = np.concatenate((memeFeature[d],noMemeFeature[d],stickerFeature[d]))
    f = np.squeeze(f)
    features[d] = f[shuffleIndices]

labels = np.concatenate((labelMeme,labelNoMeme,labelSticker))
labels = np.squeeze(labels)
labels = labels[shuffleIndices]
fname = np.concatenate((fnameMeme, fnameNoMeme, fnameSticker))
fname = np.squeeze(fname)
fname = fname[shuffleIndices]

    #train test split
split_ratio = 0.8
t = int(samples*3*split_ratio)
train_features = {}
test_features = {}
train_labels = labels[:t]
test_labels = labels[t:]
fname_train = fname[:t]
fname_test = fname[t:]

for d in desc:
    train_features[d] = features[d][:t]
    test_features[d] = features[d][t:]



missclassifiedMeme = {}
missclassifiedNomeme = {}
missclassifiedSticker = {}
for d in desc:
    classifiers = [RandomForestClassifier(n_estimators=100)]
    classifiers.append(GaussianNB())
    classifiers.append(LogisticRegression(max_iter=500, solver="sag"))
    classifiers.append(MLPClassifier(**mlpparams[d]))
    classifiers.append(SVC(**svcparams[d]))
    classifiers.append(KNeighborsClassifier(**knnparams[d]))
    for clf in classifiers:
        clf.fit(train_features[d],train_labels)
        clfpredictions = clf.predict(test_features[d])
        for i in range(len(test_features[d])):
            if test_labels[i] == 1 and clfpredictions[i] != 1:
                if fname_test[i] not in missclassifiedMeme:
                    missclassifiedMeme[fname_test[i]] = [(d,type(clf))]
                else:
                    missclassifiedMeme[fname_test[i]].append((d,type(clf)))
            if test_labels[i] == 0 and clfpredictions[i] != 0:
                if fname_test[i] not in missclassifiedNomeme:
                    missclassifiedNomeme[fname_test[i]] = [(d,type(clf))]
                else:
                    missclassifiedNomeme[fname_test[i]].append((d,type(clf)))
            if test_labels[i] == 2 and clfpredictions[i] != 2:
                if fname_test[i] not in missclassifiedSticker:
                    missclassifiedSticker[fname_test[i]] = [(d,type(clf))]
                else:
                    missclassifiedSticker[fname_test[i]].append((d,type(clf)))
                    
#        print(d, type(clf))
#        print("accuracy score: %.3f" % (accuracy_score(test_labels,clfpredictions)))
#        print("precision score: %.3f" % (precision_score(test_labels,clfpredictions, average="macro")))
#        print("recall score: %.3f" % (recall_score(test_labels,clfpredictions, average="macro")))
#        print("f1 score: %.3f" % (f1_score(test_labels,clfpredictions, average="macro")))
    

tmemes = len([i for i in test_labels if i == 1])
tnmemes = len([i for i in test_labels if i == 0])
tsmemes = len([i for i in test_labels if i == 2])

print("Total memes en test:",tmemes)
print("Total no memes en test:",tnmemes)
print("Total stickers memes en test:",tsmemes)

#histMemes = [len(missclassifiedMeme[i]) for i in missclassifiedMeme]
#histMemes = {i:histMemes.count(i) for i in histMemes}
#print(histMemes)
#histNoMemes = [len(missclassifiedNomeme[i]) for i in missclassifiedNomeme]
#histNoMemes = {i:histNoMemes.count(i) for i in histNoMemes}
#print(histNoMemes)
#
#histSticker = [len(missclassifiedSticker[i]) for i in missclassifiedSticker]
#histSticker = {i:histSticker.count(i) for i in histSticker}
#print(histSticker)


missclassifiedMeme =[i.decode('utf-8') for i in missclassifiedMeme if len(missclassifiedMeme[i]) == len(desc)*6]

print("Memes que ninguna combinación pudo clasificar correctamente:", len(missclassifiedMeme))
print("Porcentaje memes difíciles: %.3f" % (100.0*len(missclassifiedMeme)/tmemes))
print("Memes difíciles:", missclassifiedMeme)

missclassifiedNomeme =[i.decode('utf-8') for i in missclassifiedNomeme if len(missclassifiedNomeme[i]) == len(desc)*6]

print("No memes que ninguna combinación pudo clasificar correctamente:", len(missclassifiedNomeme))
print("Porcentaje no memes difíciles: %.3f" % (100.0*len(missclassifiedNomeme)/tnmemes))
print("No memes difíciles:", missclassifiedNomeme)


missclassifiedSticker =[i.decode('utf-8') for i in missclassifiedSticker if len(missclassifiedSticker[i]) == len(desc)*6]

print("Stickers que ninguna combinación pudo clasificar correctamente:", len(missclassifiedSticker))
print("Porcentaje stickers difíciles: %.3f" % (100.0*len(missclassifiedSticker)/tsmemes))
print("Stickers difíciles:", missclassifiedSticker)

with open("memes-dificiles.txt",'a') as f:
    for i in missclassifiedMeme:
        f.write(i)
        f.write('\n')
    
with open("no-memes-dificiles.txt",'a') as f:
    for i in missclassifiedNomeme:
        f.write(i)
        f.write('\n')
    
with open("stickers-dificiles.txt",'a') as f:
    for i in missclassifiedSticker:
        f.write(i)
        f.write('\n')
    