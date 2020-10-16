# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:42:52 2020

@author: albsa
"""


import cv2
import glob
import numpy as np
from ColorLayoutComputer import ColorLayoutComputer
from GrayHistogramComputer import GrayHistogramComputer
from EdgeHistogramComputer import EdgeHistogramComputer
from OrientedGradientsComputer import OrientedGradientsComputer
from DeCAF7Computer import DeCAF7Computer
from ResNet152Computer import ResNet152Computer
import h5py

computer = []
computer.append(GrayHistogramComputer(8,8,32))
computer.append(ColorLayoutComputer())
computer.append(EdgeHistogramComputer(2,2))
computer.append(OrientedGradientsComputer(2,2,1))
computer.append(DeCAF7Computer())
computer.append(ResNet152Computer())
computerNames = ["gray","color","edge","hog","decaf", "resnet"]

memedir = ".\\data\\MemesDataSet\\Meme"
nomemedir = ".\\data\\MemesDataSet\\No-meme"
stickerdir = ".\\data\\MemesDataSet\\Sticker"
      
memes = glob.glob(memedir + "\\*.jpg")
noMemes = glob.glob(nomemedir + "\\*.jpg")
stickers = glob.glob(stickerdir + "\\*.jpg")


noneMeme = 0
noneSticker = 0
noneNoMeme = 0
for i in range(len(computerNames)):
    memeFile = computerNames[i] + "-meme.h5"
    noMemeFile = computerNames[i] + "-no-meme.h5"
    stickerFile = computerNames[i] + "-sticker.h5"
    
    p = [(memes,memeFile),(noMemes,noMemeFile),(stickers,stickerFile)]
    for flist, ffile in p:    
        print("calculating ", ffile)
        features = []
        names = []
        cnt = 0
        for name in flist:
            print(100.0*cnt/len(flist), ffile, name)
            img = cv2.imread(name)
            if img is None:
                print("missing image - skipping")
                continue
            imgDesc = computer[i].compute(img)
            features.append(imgDesc)
            names.append(name[-15:])
            cnt += 1
        features = np.stack(features)
        names = np.asarray(names,dtype="S")
        with h5py.File(ffile, 'w') as hf:
            hf.create_dataset("features",  data=features, compression='gzip', compression_opts=9)
            hf.create_dataset("filename",  data=names, compression='gzip', compression_opts=9)    



