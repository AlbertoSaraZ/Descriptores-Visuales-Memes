# -*- coding: utf-8 -*-
"""
Created on Thu May  7 02:51:38 2020

@author: albsa
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import glob
from random import shuffle
from torch.utils import data as dt

from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score, f1_score, recall_score, precision_score

class Dataset(dt.Dataset):
  def __init__(self, img, labels, fnames):
        self.labels = labels
        self.img = img
        self.fnames = fnames
  def __len__(self):
        return len(self.img)

  def __getitem__(self, index):
        X = self.img[index]
        y = self.labels[index]
        f = self.fnames[index]
        return X, y, f
    
resnet152 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)#models.resnet152(pretrained=True)
for p in resnet152.parameters():
    p.requires_grad = False

resnet152.fc = nn.Linear(2048,3)        

optimizer = optim.SGD(resnet152.fc.parameters(), lr=0.0001, momentum=0.9)
criterion = nn.CrossEntropyLoss()



resnet152.to('cuda')
memedir = ".\\data\\MemesDataSet\\Meme"
nomemedir = ".\\data\\MemesDataSet\\No-meme"
stickerdir = ".\\data\\MemesDataSet\\Sticker"

memes = glob.glob(memedir + "\\*.jpg")
shuffle(memes)
memes = memes
noMemes = glob.glob(nomemedir + "\\*.jpg")
shuffle(noMemes)
noMemes = noMemes[:len(memes)]
stickers = glob.glob(stickerdir + "\\*.jpg")
shuffle(stickers)
stickers = stickers[:len(memes)]

memeArray = []
noMemeArray = []
stickerArray = []
label=[]
i = 0
imgfname = []
for group, arr in ((noMemes, noMemeArray),(memes, memeArray), (stickers, stickerArray)):
    for name in group:
        img = cv2.imread(name)
        if img is None:
            print("missing image - skipping")
            continue
        imgfname.append(int(name[-11:-4]))
        img = cv2.resize(img, (224,224))
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        arr.append(input_tensor)
        label.append(i)
    i+=1


data = memeArray + noMemeArray + stickerArray
data = torch.stack(data)
label = torch.tensor(label)


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
imgfname = np.asarray(imgfname)
imgfname = imgfname[indices]
data = data[indices]
label = label[indices]
training = data[:int(len(data)*0.8)]
test = data[len(training):]
training_labels = label[:int(len(label)*0.8)]
test_labels = label[len(training_labels):]


training_fname = imgfname[:int(len(label)*0.8)]
test_fname = imgfname[len(training_labels):]

training_dset = Dataset(training, training_labels, training_fname)
test_dset = Dataset(test, test_labels, test_fname)

trainloader = dt.DataLoader(training_dset, batch_size=16, shuffle=True, num_workers=0)
testloader = dt.DataLoader(test_dset, batch_size=200, shuffle=True, num_workers=0)

lsscount = 0
epsilon = 0.0001
consec_epochs = 5
running_loss = 0.0

train_losses = []
test_losses = []
test_accuracy = []
for epoch in range(50): 
    last_loss = running_loss
    running_loss = 0.0
    test_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to('cuda'), data[1].to('cuda')
        optimizer.zero_grad()

        outputs = resnet152(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss / len(trainloader)))

    train_losses.append(running_loss / len(trainloader))
    correct = 0
    total = 0
    running_loss /= len(training)
    with torch.no_grad():
#        resnet152.load_state_dict(torch.load('./resnet152.pth'))
        for data in testloader:
            images, labels = data[0].to('cuda'), data[1].to('cuda')
            outputs = resnet152(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('[%d] testing loss: %.3f' %
          (epoch + 1, test_loss / len(testloader)))
    test_losses.append(test_loss / len(testloader))
    test_loss = 0.0
    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))
    test_accuracy.append(correct / total)
    if np.abs(last_loss - running_loss) < epsilon:
        lsscount += 1
        if lsscount == consec_epochs:     
            print('Stopping: loss difference smaller than ',epsilon," for ", consec_epochs, " consecutive epochs")
            break
    else:
        lsscount = 0


#torch.save(resnet152.state_dict(), './resnet152.pth')
        
confusion_matrix = torch.zeros(3,3)
target_true = 0
predicted_true = 0
correct_true = 0
preds = []
Y = []
filenamesmeme = []
filenamesnomeme = []
filenamessticker = []

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to('cuda'), data[1].to('cuda')
        outputs = resnet152(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        pred = torch.argmax(outputs, 1)
        preds.append(pred.cpu().numpy())
        Y.append(labels.cpu().numpy())
        for t, p, f in zip(labels.view(-1), pred.view(-1), data[2]):
                confusion_matrix[t.long(), p.long()] += 1
                if t.long() != p.long() and t.long() == 1:
                    file_name = '0000000' + str(f.item()) + '.jpg'
                    file_name = file_name[-11:]
                    file_name = 'img_' + file_name
                    filenamesmeme.append((file_name))
                elif t.long() != p.long() and t.long() == 0:
                    file_name = '0000000' + str(f.item()) + '.jpg'
                    file_name = file_name[-11:]
                    file_name = 'img_' + file_name
                    filenamesnomeme.append((file_name))      
                elif t.long() != p.long() and t.long() == 2:
                    file_name = '0000000' + str(f.item()) + '.jpg'
                    file_name = file_name[-11:]
                    file_name = 'img_' + file_name
                    filenamessticker.append((file_name))     

print(filenamesmeme)
print(filenamesnomeme)
print(filenamessticker)

preds = np.concatenate(preds)
Y = np.concatenate(Y)
confusion_matrix = confusion_matrix.numpy()
confusion_matrix = confusion_matrix.astype('int32')
accuracy = accuracy_score(preds, Y)
recall = recall_score(preds, Y, average = 'macro')
precision = precision_score(preds, Y, average = 'macro')
f1_score = f1_score(preds, Y, average = 'macro')

print('accuracy: ',accuracy)
print('precision: ',precision)
print('recall: ',recall)
print('f1_score: ',f1_score)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                              display_labels=['No Meme','Meme', 'Sticker'])

disp = disp.plot(cmap='Blues',values_format ="d")
plt.show()

cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['No Meme','Meme', 'Sticker'])

disp = disp.plot(cmap='Blues')
plt.show()
f1 = plt.figure(1)
ax1 = f1.add_subplot(111);
ax1.set_title('Pérdida durante entrenamiento')    
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Pérdida')
ax1.plot(train_losses, c='r', label='Entrenamiento')
ax1.plot(test_losses, c='b', label='Prueba')
legend = ax1.legend()

f2 = plt.figure(2)
ax2 = f2.add_subplot(111);
ax2.set_title('Precisión durante entrenamiento')    
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Precisión')
ax2.plot(test_accuracy, c='b')
plt.show()