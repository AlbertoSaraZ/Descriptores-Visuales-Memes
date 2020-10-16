# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:27:41 2020

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
import json
import h5py


parser = argparse.ArgumentParser()


parser.add_argument("-d", "--descriptor", help="Descriptor, puede ser gray color edge hog decaf \n default gray")
parser.add_argument("-mdir", "--memedirectory", help="Directorio con imagenes de memes default .\\data\\MemesDataSet\\Meme")
parser.add_argument("-nmdir", "--nomemedirectory", help="Directorio con imagenes de no memes default .\\data\\MemesDataSet\\No-meme")
parser.add_argument("-s", "--sticker", help="Cargar stickers, poner flag para usar stickers en las clases", action="store_true")
parser.add_argument("-cv", "--crossvalidation", help="Iteraciones para validación cruzada default 3",default=3, type=int)
args = parser.parse_args()

computer = ""

if args.descriptor == "gray":
    descriptorName = "Gray Histogram"
    #0.47 sec
elif args.descriptor == "color":
    descriptorName = "Color Layout"
    #0.0033 sec
elif args.descriptor == "edge":
    descriptorName = "Edge Histogram"
    #0.021 sec
elif args.descriptor == "hog":
    descriptorName = "Histogram of Oriented Gradients"
    #0.025 sec
elif args.descriptor == "decaf":
    descriptorName = "DeCAF7"
    #0.8 sec
elif args.descriptor == "resnet":
    descriptorName = "ResNet152"
else:
    print("No se especificó descriptor, utilizando histograma de gris por defecto")
    descriptorName = "Gray Histogram"
    args.descriptor = "gray"
    
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
    
      
memes = glob.glob(memedir + "\\*.jpg")
noMemes = glob.glob(nomemedir + "\\*.jpg")




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
labelSticker = np.full((stickerFeature.shape[0],1),2)

features = np.concatenate((noMemeFeature, memeFeature,stickerFeature))
labels = np.concatenate((labelNoMeme, labelMeme,labelSticker))

features = np.squeeze(features)
labels = np.squeeze(labels)
undersampler = RandomUnderSampler(random_state=0)
features, labels = undersampler.fit_resample(features, labels)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 0)

scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_features)
train_features = scaling.transform(train_features)
test_features = scaling.transform(test_features)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

    

parameter_space = []
parameter_space.append({
    'hidden_layer_sizes': [(3000),(500,10)],
    'activation': ['tanh'],
    'solver': ['sgd'],
    'alpha': [0.0001],
    'learning_rate': ['adaptive','invscaling'],
    'early_stopping': [True],
})
parameter_space.append({
    "algorithm": ["ball_tree","kd_tree"],
    "p": [2,3,7],
    "n_neighbors": [3,12,20],
})   
parameter_space.append({
    "kernel": ["rbf",'poly'],
    "degree": [1,3,5],
    "C": [0.1,1,10],
    "coef0": [0,1],
})

classifier = [MLPClassifier(max_iter=500), KNeighborsClassifier(), SVC(max_iter=500)]   


for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], parameter_space[i], n_jobs=-1, cv=args.crossvalidation, verbose=1)
    clf.fit(train_features,np.ravel(train_labels))
    clfPred = clf.predict(test_features)
    clfP = precision_score(test_labels, clfPred, average="macro")
    clfR = recall_score(test_labels, clfPred, average="macro")
    clfA = accuracy_score(test_labels, clfPred)
    clfF = f1_score(test_labels, clfPred, average="macro")
    print('Best parameters found:\n', clf.best_params_)
    print("accuracy",100*clfA,"precision",100*clfP,"recall",100*clfR,"f1",100*clfF)